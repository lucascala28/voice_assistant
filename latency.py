"""Dependency-free per-turn latency marks. Enabled with env LATENCY_LOG=1, otherwise every call is a no-op."""
import os
import statistics
import threading
import time

ENABLED = os.getenv("LATENCY_LOG") == "1"

_lock = threading.Lock()
_marks = {}      # name -> perf_counter timestamp (first write wins unless overwrite=True)
_content_ts = []  # arrival times of SSE chunks that carried content
_sse_count = 0

# Display order for the summary table (marks not recorded this turn are skipped)
ORDER = [
    ("speech_end", "user stopped speaking (derived)"),
    ("endpoint", "VAD endpoint detected"),
    ("transcribe_done", "transcription done"),
    ("llm_request_sent", "LLM request sent"),
    ("first_sse_chunk", "first SSE chunk"),
    ("first_content_token", "first content token"),
    ("first_chunk_ready", "first sentence/chunk ready"),
    ("first_tts_audio_ready", "first TTS audio ready"),
    ("audio_playing", "audio playing"),
    ("total", "turn finished (text + queued)"),
]


# Start a fresh turn: clear all marks and SSE stats.
def new_turn():
    global _sse_count
    if not ENABLED:
        return
    with _lock:
        _marks.clear()
        _content_ts.clear()
        _sse_count = 0


# Record a timestamp for `name`; only the first call per turn counts unless overwrite=True.
def mark(name, t=None, overwrite=False):
    if not ENABLED:
        return
    t = time.perf_counter() if t is None else t
    with _lock:
        if overwrite or name not in _marks:
            _marks[name] = t


# Count one SSE chunk, noting its arrival time if it carried spoken content.
def note_sse(has_content):
    global _sse_count
    if not ENABLED:
        return
    t = time.perf_counter()
    with _lock:
        _sse_count += 1
        if has_content:
            _content_ts.append(t)


# Summarise inter-token gaps so we can tell if the server streams or buffers content.
def sse_summary():
    with _lock:
        ts = list(_content_ts)
        n = _sse_count
    if len(ts) < 2:
        return f"{n} SSE chunks, {len(ts)} with content"
    gaps = [(b - a) * 1000 for a, b in zip(ts, ts[1:])]
    span = (ts[-1] - ts[0]) * 1000
    note = ""
    if len(ts) >= 8 and span < 50:
        note = "  <- content arrived in a burst: server likely buffers"
    return (f"{n} SSE chunks, {len(ts)} with content, content span {span:.0f}ms, "
            f"gap median {statistics.median(gaps):.1f}ms max {max(gaps):.1f}ms{note}")


# Print the per-turn table (ms relative to end of speech, plus delta from previous row).
def report():
    if not ENABLED:
        return
    mark("total")
    with _lock:
        marks = dict(_marks)
    if "speech_end" not in marks:
        return
    base = marks["speech_end"]
    lines = ["", "---- latency (ms from end of speech) ----"]
    prev = base
    for key, label in ORDER:
        if key in marks:
            t = marks[key]
            lines.append(f"{label:<34}{(t - base) * 1000:>8.0f}  (+{(t - prev) * 1000:.0f})")
            prev = max(prev, t)
    if "audio_playing" in marks:
        lines.append(f"{'SPEECH END -> FIRST AUDIO':<34}{(marks['audio_playing'] - base) * 1000:>8.0f}")
    lines.append(sse_summary())
    print("\n".join(lines) + "\n")
