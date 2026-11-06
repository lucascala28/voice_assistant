"""send_to_model with tts / smart_home / playsound3 / weather stubbed and requests.post patched."""
import asyncio
import importlib
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


# Build an SSE line (bytes) from a delta dict.
def sse(delta):
    return b"data: " + json.dumps({"choices": [{"delta": delta}]}).encode()


class FakeResponse:
    def __init__(self, lines, status_code=200):
        self.lines = lines
        self.status_code = status_code
        self.closed = False

    def close(self):
        self.closed = True

    def iter_lines(self, chunk_size=None):
        return iter(self.lines)


@pytest.fixture
def rtr(monkeypatch, tmp_path):
    spoken, posts = [], []
    weather_calls = []

    class DeviceNotFoundError(Exception):
        pass

    async def control_device(device, action):
        if device == "ghost":
            raise DeviceNotFoundError()
        if device == "offline":
            raise OSError("plug unreachable")

    tts = types.ModuleType("tts")
    tts.enqueue_tts = spoken.append
    smart_home = types.ModuleType("smart_home")
    smart_home.control_device, smart_home.DeviceNotFoundError = control_device, DeviceNotFoundError
    playsound3 = types.ModuleType("playsound3")
    playsound3.playsound = lambda *a, **k: None
    weather = types.ModuleType("weather")

    def get_weather(location, detailed):
        weather_calls.append((location, detailed))
        if location == "Atlantis":
            raise ValueError("Location not found")
        return ("clear sky", 72.5, 70, 75, 60, 5, 0)

    weather.get_weather = get_weather
    for mod in (tts, smart_home, playsound3, weather):
        monkeypatch.setitem(sys.modules, mod.__name__, mod)

    # Keep the test hermetic: temp devices file, no real .env, fixed env values
    devices = tmp_path / "devices.json"
    devices.write_text("{}")
    monkeypatch.setenv("DEVICES_FILE", str(devices))
    monkeypatch.setenv("LLM_MAX_TOKENS", "300")
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    sys.modules.pop("req_to_res", None)
    mod = importlib.import_module("req_to_res")
    mod.context.clear()

    # Each post() call pops the next canned response
    responses = []

    def fake_post(url, json=None, **kw):
        posts.append(json)
        item = responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item if isinstance(item, FakeResponse) else FakeResponse(item)

    monkeypatch.setattr(mod.requests, "post", fake_post)
    yield types.SimpleNamespace(mod=mod, spoken=spoken, posts=posts, responses=responses, weather_calls=weather_calls)
    sys.modules.pop("req_to_res", None)


def test_plain_reply_is_chunked_while_streaming(rtr):
    rtr.responses.append([
        sse({"role": "assistant"}), sse({"content": "Sure thing"}), sse({"content": ".\n"}),
        sse({"content": "It is 72"}), sse({"content": "."}), sse({"content": "5 out"}), sse({"content": "."}),
        b"data: [DONE]",
    ])
    reply = asyncio.run(rtr.mod.send_to_model("hello"))
    assert rtr.spoken == ["Sure thing.", "It is 72.5 out."]
    assert reply == "Sure thing.\nIt is 72.5 out."
    assert rtr.posts[0]["max_tokens"] == 300 and rtr.posts[0]["messages"][0]["role"] == "system"


def test_weather_tool_followup(rtr):
    rtr.responses.append([
        sse({"role": "assistant", "tool_calls": [{"index": 0, "id": "t1", "function": {"name": "get_weather", "arguments": '{"location": "Boston"}'}}]}),
        b"data: [DONE]",
    ])
    rtr.responses.append([sse({"content": "It is 72 degrees."}), b"data: [DONE]"])
    asyncio.run(rtr.mod.send_to_model("weather in Boston?"))
    assert rtr.weather_calls == [("Boston", False)]
    assert rtr.spoken == ["It is 72 degrees."]
    roles = [m["role"] for m in rtr.mod.context]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert "72.5" in rtr.mod.context[2]["content"]            # real tool result, not the spoken text
    follow = rtr.posts[1]["messages"]
    assert [m["role"] for m in follow] == ["system", "user", "assistant", "tool"]
    assert sum(1 for m in follow if m.get("tool_calls")) == 1  # tool call appears once


def test_weather_location_not_found_is_handled(rtr):
    rtr.responses.append([
        sse({"tool_calls": [{"index": 0, "id": "t2", "function": {"name": "get_weather", "arguments": '{"location": "Atlantis"}'}}]}),
        b"data: [DONE]",
    ])
    asyncio.run(rtr.mod.send_to_model("weather in Atlantis"))
    assert rtr.spoken == ["I couldn't find that location"]
    assert rtr.mod.context[-1]["role"] == "tool" and len(rtr.posts) == 1


def test_device_not_found_and_bad_json(rtr):
    rtr.responses.append([
        sse({"tool_calls": [{"index": 0, "id": "t3", "function": {"name": "control_device", "arguments": '{"device": "ghost", "action": "on"}'}}]}),
    ])
    asyncio.run(rtr.mod.send_to_model("turn on ghost"))
    rtr.responses.append([
        sse({"tool_calls": [{"index": 0, "id": "t4", "function": {"name": "control_device", "arguments": "{bad"}}]}),
    ])
    asyncio.run(rtr.mod.send_to_model("again"))
    assert rtr.spoken == ["Device not found amongst known devices", "There was an error calling the tool"]


def test_trim_context_starts_at_user(rtr):
    ctx = [{"role": "user"}, {"role": "assistant", "tool_calls": [1]}, {"role": "tool"}, {"role": "assistant"}] * 12
    out = rtr.mod.trim_context(ctx, limit=40)
    assert len(out) <= 40 and out[0]["role"] == "user"
    assert rtr.mod.trim_context([{"role": "tool"}, {"role": "user"}], limit=40)[0]["role"] == "user"


def test_tool_schema_required_inside_parameters(rtr):
    weather = next(t for t in rtr.mod.tools if t["function"]["name"] == "get_weather")["function"]
    assert "required" not in weather and weather["parameters"]["required"] == ["location"]


# Canned response containing one tool call.
def tool_resp(name, args, call_id="x"):
    return [sse({"tool_calls": [{"index": 0, "id": call_id, "function": {"name": name, "arguments": args}}]}), b"data: [DONE]"]


def test_text_before_tool_call_is_stored_with_tool_calls(rtr):
    rtr.responses.append([sse({"content": "Checking now."}),
                          sse({"tool_calls": [{"index": 0, "id": "p1", "function": {"name": "control_device", "arguments": '{"device": "lamp", "action": "on"}'}}]})])
    asyncio.run(rtr.mod.send_to_model("lamp on"))
    assert rtr.spoken == ["Checking now."]
    assistant = rtr.mod.context[1]
    assert assistant["content"] == "Checking now." and assistant["tool_calls"][0]["id"] == "p1"
    assert [m["role"] for m in rtr.mod.context] == ["user", "assistant", "tool"]


def test_unexpected_tool_exception_still_one_tool_message(rtr):
    rtr.responses.append(tool_resp("control_device", '{"device": "offline", "action": "on"}'))
    asyncio.run(rtr.mod.send_to_model("offline plug"))
    assert [m["role"] for m in rtr.mod.context] == ["user", "assistant", "tool"]
    assert rtr.spoken == ["There was an error calling the tool"]


def test_null_args_and_unknown_tool(rtr):
    rtr.responses.append(tool_resp("control_device", "null"))
    asyncio.run(rtr.mod.send_to_model("a"))
    rtr.responses.append(tool_resp("", "{}"))
    asyncio.run(rtr.mod.send_to_model("b"))
    assert rtr.spoken == ["There was an error calling the tool", "I don't know how to do that"]
    assert [m["role"] for m in rtr.mod.context] == ["user", "assistant", "tool"] * 2


def test_post_failure_speaks_fallback_and_keeps_context_valid(rtr):
    import requests
    rtr.responses.append(requests.exceptions.ConnectionError("down"))
    asyncio.run(rtr.mod.send_to_model("hello"))
    assert rtr.spoken == ["I can't reach the model right now"]
    assert rtr.mod.context == []


def test_stream_failure_mid_response(rtr):
    import requests

    class Broken(FakeResponse):
        def iter_lines(self, chunk_size=None):
            yield sse({"content": "Hi"})
            raise requests.exceptions.ChunkedEncodingError("cut")

    rtr.responses.append(Broken([]))
    asyncio.run(rtr.mod.send_to_model("hello"))
    assert rtr.spoken == ["I can't reach the model right now"]
    assert rtr.mod.context == []


def test_non_200_and_sse_error(rtr):
    rtr.responses.append(FakeResponse([], status_code=503))
    asyncio.run(rtr.mod.send_to_model("hello"))
    rtr.responses.append([b'data: {"error": {"message": "model crashed"}}'])
    asyncio.run(rtr.mod.send_to_model("again"))
    assert rtr.spoken == ["I can't reach the model right now"] * 2


def test_followup_failure_keeps_tool_message(rtr):
    import requests
    rtr.responses.append(tool_resp("get_weather", '{"location": "Boston"}'))
    rtr.responses.append(requests.exceptions.ReadTimeout("slow"))
    asyncio.run(rtr.mod.send_to_model("weather"))
    assert rtr.spoken == ["I can't reach the model right now"]
    assert [m["role"] for m in rtr.mod.context] == ["user", "assistant", "tool"]


def test_multiple_tool_calls_one_tool_message_each(rtr):
    rtr.responses.append([
        sse({"tool_calls": [{"index": 0, "id": "a", "function": {"name": "control_device", "arguments": '{"device": "lamp", "action": "on"}'}}]}),
        sse({"tool_calls": [{"index": 1, "id": "b", "function": {"name": "control_device", "arguments": '{"device": "ghost", "action": "on"}'}}]}),
    ])
    asyncio.run(rtr.mod.send_to_model("both"))
    ctx = rtr.mod.context
    assert [m["role"] for m in ctx] == ["user", "assistant", "tool", "tool"]
    assert len(ctx[1]["tool_calls"]) == 2 and [m["tool_call_id"] for m in ctx[2:]] == ["a", "b"]


def test_followup_sends_full_context_prefix(rtr):
    rtr.responses.append(tool_resp("get_weather", '{"location": "Boston"}'))
    rtr.responses.append([sse({"content": "Sunny."}), b"data: [DONE]"])
    asyncio.run(rtr.mod.send_to_model("weather"))
    first, second = rtr.posts[0]["messages"], rtr.posts[1]["messages"]
    assert second[:len(first)] == first and len(second) == len(first) + 2
    assert rtr.posts[1]["tools"] == rtr.posts[0]["tools"]       # same tools => same rendered prompt prefix


def test_think_block_not_spoken(rtr):
    rtr.responses.append([sse({"content": "<thi"}), sse({"content": "nk>secret plan.</th"}), sse({"content": "ink>Hello there."}), b"data: [DONE]"])
    reply = asyncio.run(rtr.mod.send_to_model("hi"))
    assert rtr.spoken == ["Hello there."] and reply == "Hello there."


def test_empty_visible_reply_pops_user_turn(rtr):
    rtr.responses.append([sse({"content": "<think>still thinking and never closes"}), b"data: [DONE]"])
    asyncio.run(rtr.mod.send_to_model("hmm"))
    assert rtr.spoken == ["Sorry, I don't have an answer for that"]
    assert rtr.mod.context == []


def test_tts_failure_does_not_escape(rtr, monkeypatch):
    def boom(text):
        raise RuntimeError("kokoro died")

    monkeypatch.setattr(rtr.mod, "enqueue_tts", boom)
    rtr.responses.append(tool_resp("control_device", '{"device": "ghost", "action": "on"}'))
    asyncio.run(rtr.mod.send_to_model("ghost"))
    assert [m["role"] for m in rtr.mod.context] == ["user", "assistant", "tool"]


def test_response_closed_on_non_200_and_after_done(rtr):
    bad = FakeResponse([], status_code=500)
    rtr.responses.append(bad)
    asyncio.run(rtr.mod.send_to_model("x"))
    good = FakeResponse([sse({"content": "Hi."}), b"data: [DONE]"])
    rtr.responses.append(good)
    asyncio.run(rtr.mod.send_to_model("y"))
    assert bad.closed and good.closed


def test_empty_tool_call_id_gets_fallback_in_messages(rtr):
    rtr.responses.append(tool_resp("control_device", '{"device": "lamp", "action": "on"}', call_id=""))
    asyncio.run(rtr.mod.send_to_model("lamp"))
    ctx = rtr.mod.context
    assert ctx[1]["tool_calls"][0]["id"] == "call_0" and ctx[2]["tool_call_id"] == "call_0"
