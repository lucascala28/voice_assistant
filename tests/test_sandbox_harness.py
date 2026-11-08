import queue
import threading
import time
from types import SimpleNamespace

import pytest

import sandbox.harness as harness
from sandbox.harness import Sandbox, check, context_problems


# Build a fake sandbox object holding just what check() reads.
def make_sb(context=(), spoken=("Hello",), chimes=(), actions=(), llm_errors=(), audio=(1,), no_tts_synth=False):
    return SimpleNamespace(
        rr=SimpleNamespace(context=list(context)), spoken=list(spoken), chimes=list(chimes),
        device_actions=list(actions), llm_errors=list(llm_errors), tts_audio=list(audio),
        args=SimpleNamespace(no_tts_synth=no_tts_synth))


# Assistant message with one tool call.
def call(name, id_, args="{}"):
    return {"role": "assistant", "tool_calls": [{"id": id_, "function": {"name": name, "arguments": args}}]}


# Tool reply message for a call id.
def reply(id_):
    return {"role": "tool", "tool_call_id": id_, "content": "ok"}


# Minimal scenario record as check() reads it.
def rec(tools=(), transcript=None):
    return {"tool_calls": [{"name": n, "args": a} for n, a in tools], "transcript": transcript}


# Well-formed contexts produce no problems.
def test_context_ok():
    ctx = [{"role": "user"}, call("a", "1"), reply("1"), {"role": "assistant"}]
    assert context_problems(ctx) == []
    assert context_problems([]) == []


# A context starting with an assistant message is flagged.
def test_context_must_start_with_user():
    assert context_problems([{"role": "assistant"}]) == ["context does not start with a user message"]


# A tool call without a reply is flagged.
def test_context_unanswered_tool_call():
    problems = context_problems([{"role": "user"}, call("a", "1")])
    assert len(problems) == 1 and "not answered" in problems[0]


# Replies must follow the order of the calls.
def test_context_tool_replies_out_of_order():
    ctx = [{"role": "user"}, {"role": "assistant", "tool_calls": [
        {"id": "1", "function": {}}, {"id": "2", "function": {}}]}, reply("2"), reply("1")]
    assert len(context_problems(ctx)) == 1


# A normal run with no expectations has no errors.
def test_check_clean_run():
    assert check({}, rec(), make_sb()) == []


# A chime alone counts as feedback; silence does not.
def test_check_no_feedback_but_chime_is_fine():
    assert check({}, rec(), make_sb(spoken=[], chimes=[1], audio=[])) == []
    assert "no feedback" in check({}, rec(), make_sb(spoken=[" "], audio=[]))[0]


# Spoken text with no captured audio is an error unless synthesis is off.
def test_check_tts_without_audio():
    assert check({}, rec(), make_sb(audio=[])) == ["TTS produced no audio"]
    assert check({}, rec(), make_sb(audio=[], no_tts_synth=True)) == []


# LLM errors recorded during the turn fail the scenario.
def test_check_llm_error_reported():
    assert "LLM error" in check({}, rec(), make_sb(llm_errors=["boom"]))[0]


# Expected tool / no tool is compared with the calls made.
def test_check_tool_expectations():
    r = rec([("get_weather", "{}")])
    assert check({"tool": "get_weather"}, r, make_sb()) == []
    assert "expected tool control_device" in check({"tool": "control_device"}, r, make_sb())[0]
    assert "expected no tool call" in check({"tool": None}, r, make_sb())[0]
    assert check({"tool": None}, rec(), make_sb()) == []


# Recorded device actions must equal the expected ones.
def test_check_device_actions():
    acts = [{"device": "lamp", "action": "on"}]
    assert check({"device_actions": acts}, rec(), make_sb(actions=acts)) == []
    assert "device actions" in check({"device_actions": acts}, rec(), make_sb())[0]


# The weather call's location must mention the expected text.
def test_check_location():
    r = rec([("get_weather", '{"location": "Paris, FR"}')])
    assert check({"location_contains": "paris"}, r, make_sb()) == []
    assert "location" in check({"location_contains": "rome"}, r, make_sb())[0]


# Transcript and reply keywords are matched case-insensitively.
def test_check_transcript_and_spoken():
    r = rec(transcript="Turn on the Lamp")
    assert check({"transcript_contains": ["lamp"]}, r, make_sb()) == []
    assert "missing 'fan'" in check({"transcript_contains": ["fan"]}, r, make_sb())[0]
    assert check({"spoken_any": ["hello"]}, rec(), make_sb()) == []
    assert "none of" in check({"spoken_any": ["rain"]}, rec(), make_sb())[0]


# Fake tts module: a real queue and a worker thread that plays each clip for `play` seconds (or crashes if `die`).
def make_tts(play=0.0, die=False):
    tts = SimpleNamespace(_audio_queue=queue.Queue(), _stop=threading.Event(), played=[], starts=0)

    def worker():
        while not tts._stop.is_set():
            try:
                clip = tts._audio_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if die:
                raise RuntimeError("worker crash")
            time.sleep(play)
            tts.played.append(clip)
            tts._audio_queue.task_done()

    def start_tts():
        tts.starts += 1
        tts._stop.clear()
        tts._worker_thread = threading.Thread(target=worker, daemon=True)
        tts._worker_thread.start()

    def stop_tts():
        tts._stop.set()
        while not tts._audio_queue.empty():
            tts._audio_queue.get_nowait()
            tts._audio_queue.task_done()
        tts._worker_thread.join(timeout=2)

    tts.start_tts, tts.stop_tts = start_tts, stop_tts
    start_tts()
    return tts


# Everything queued is played, then wait_playback reports no problem.
def test_wait_playback_drains():
    tts = make_tts()
    for i in range(3):
        tts._audio_queue.put(i)
    assert Sandbox.wait_playback(SimpleNamespace(tts=tts)) is None
    assert tts.played == [0, 1, 2]
    tts.stop_tts()


# A crashed worker is reported, the leftovers are dropped and a fresh worker is started.
@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")   # the crash is the point
def test_wait_playback_dead_worker_restarted():
    tts = make_tts(die=True)
    tts._audio_queue.put(1)
    tts._audio_queue.put(2)
    assert Sandbox.wait_playback(SimpleNamespace(tts=tts)) == "playback worker died"
    assert tts.starts == 2 and tts._audio_queue.unfinished_tasks == 0
    tts.stop_tts()


# A slow but live worker is reported as stalled and is healthy for the next scenario (no task_done underflow).
def test_wait_playback_stall_then_healthy(monkeypatch):
    monkeypatch.setattr(harness, "PLAYBACK_TIMEOUT", 0.3)
    tts = make_tts(play=0.6)
    for i in range(3):
        tts._audio_queue.put(i)
    assert Sandbox.wait_playback(SimpleNamespace(tts=tts)) == "playback stalled"
    assert tts._worker_thread.is_alive() and tts._audio_queue.unfinished_tasks == 0
    monkeypatch.setattr(harness, "PLAYBACK_TIMEOUT", 5)
    tts._audio_queue.put(9)
    assert Sandbox.wait_playback(SimpleNamespace(tts=tts)) is None
    assert tts.played[-1] == 9
    tts.stop_tts()
