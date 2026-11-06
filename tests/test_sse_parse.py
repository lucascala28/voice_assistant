import json

from sse_parse import parse_line, content_of, ToolCallAccumulator


# Build an SSE "data:" line (as bytes, like requests.iter_lines yields) from a delta dict.
def sse(delta):
    return b"data: " + json.dumps({"choices": [{"delta": delta}]}).encode()


# Run lines through the parser the way req_to_res does; return (text, accumulator).
def run(lines):
    acc, text = ToolCallAccumulator(), ""
    for line in lines:
        kind, delta = parse_line(line)
        if kind == "done":
            break
        if kind != "delta":
            continue
        acc.add(delta)
        text += content_of(delta)
    return text, acc


def test_role_only_first_chunk_then_text():
    text, acc = run([sse({"role": "assistant"}), sse({"content": "Hi"}), sse({"content": " there"}), b"data: [DONE]"])
    assert text == "Hi there" and not acc.all()


def test_tool_call_in_first_chunk_is_accumulated():
    lines = [
        sse({"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "get_weather", "arguments": ""}}]}),
        sse({"tool_calls": [{"index": 0, "function": {"arguments": '{"loc'}}]}),
        sse({"tool_calls": [{"index": 0, "function": {"arguments": 'ation": "Boston"}'}}]}),
        b"data: [DONE]",
    ]
    _, acc = run(lines)
    call = acc.all()[0]
    assert call == {"id": "c1", "name": "get_weather", "args": '{"location": "Boston"}'}
    assert json.loads(call["args"]) == {"location": "Boston"}


def test_text_then_tool_call_is_detected():
    lines = [
        sse({"role": "assistant"}),
        sse({"content": "Let me check. "}),
        sse({"tool_calls": [{"index": 0, "id": "c2", "function": {"name": "control_device", "arguments": "{}"}}]}),
    ]
    text, acc = run(lines)
    assert text == "Let me check. "
    assert acc.all()[0]["name"] == "control_device"


def test_null_content_and_null_arguments():
    text, acc = run([
        sse({"content": None}),
        sse({"tool_calls": [{"index": 0, "id": "c3", "function": {"name": "x", "arguments": None}}]}),
    ])
    assert text == "" and acc.all()[0]["args"] == ""


def test_done_and_noise_lines():
    assert parse_line(b"data: [DONE]") == ("done", None)
    assert parse_line(b"") == ("skip", None)
    assert parse_line(b": keepalive") == ("skip", None)
    assert parse_line(b"data: {not json") == ("skip", None)
    assert parse_line("data:[DONE]") == ("done", None)
    assert parse_line(b'data: {"choices": []}') == ("skip", None)


def test_error_chunk():
    kind, payload = parse_line(b'data: {"error": {"message": "boom"}}')
    assert kind == "error" and payload["message"] == "boom"


def test_reasoning_content_is_not_spoken():
    text, _ = run([
        sse({"reasoning_content": "thinking..."}),
        sse({"reasoning": "more"}),
        sse({"content": "Answer."}),
    ])
    assert text == "Answer."


def test_multiple_tool_calls_by_index():
    _, acc = run([
        sse({"tool_calls": [{"index": 1, "id": "b", "function": {"name": "two", "arguments": "{}"}}]}),
        sse({"tool_calls": [{"index": 0, "id": "a", "function": {"name": "one", "arguments": "{}"}}]}),
    ])
    assert [c["name"] for c in acc.all()] == ["one", "two"]


def test_finish_reason_is_surfaced():
    line = b'data: {"choices": [{"delta": {}, "finish_reason": "length"}]}'
    assert parse_line(line)[1]["finish_reason"] == "length"


def test_missing_tool_call_id_gets_fallback():
    _, acc = run([sse({"tool_calls": [{"index": 0, "id": "", "function": {"name": "x", "arguments": "{}"}}]})])
    assert acc.all()[0]["id"] == "call_0"
