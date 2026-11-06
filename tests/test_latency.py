import importlib

import pytest


# Re-import latency with LATENCY_LOG set (it reads the env at import time).
@pytest.fixture
def lat(monkeypatch):
    monkeypatch.setenv("LATENCY_LOG", "1")
    import latency
    yield importlib.reload(latency)
    monkeypatch.delenv("LATENCY_LOG", raising=False)
    importlib.reload(latency)      # restore the disabled default for other tests


def test_disabled_is_noop(monkeypatch, capsys):
    monkeypatch.delenv("LATENCY_LOG", raising=False)
    import latency
    latency = importlib.reload(latency)
    latency.new_turn()
    latency.mark("speech_end")
    latency.note_sse(True)
    latency.report()
    assert capsys.readouterr().out == "" and latency._marks == {}


def test_first_mark_wins_and_report(lat, capsys):
    lat.new_turn()
    lat.mark("speech_end", 10.0)
    lat.mark("endpoint", 10.5)
    lat.mark("audio_playing", 11.0)
    lat.mark("audio_playing", 99.0)           # ignored: first write wins
    lat.report()
    out = capsys.readouterr().out
    assert "SPEECH END -> FIRST AUDIO" in out and "1000" in out
    assert lat._marks["audio_playing"] == 11.0


def test_sse_summary_flags_burst(lat):
    lat.new_turn()
    for _ in range(10):
        lat.note_sse(True)
    assert "buffers" in lat.sse_summary()
