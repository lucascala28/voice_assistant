"""End-to-end sandbox harness for Beemo.

Runs scripted conversations through the real pipeline (LLM -> tool calls -> TTS, and optionally
Kokoro-synthesised speech -> VAD -> Whisper first) with every piece of hardware and the outside world faked:
  - sounddevice is replaced by a stub (no mic is opened; sd.play records the audio instead of playing it, so the
    real tts playback worker runs and its output is what ends up in the beemo WAVs)
  - the wake word is skipped (the stub returns immediately)
  - smart plugs are never contacted: control_device resolves against a fixture devices file and records the action
  - the weather API is canned unless --live-weather is passed
The LLM is real (LLM_HOST / LLM_MODEL), because that is the part most likely to break a real conversation.

Exit codes: 0 all scenarios passed, 1 a scenario failed, 2 the sandbox could not run (e.g. LLM unreachable).
"""
import argparse
import asyncio
import json
import os
import queue
import sys
import time
import types
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SAMPLE_RATE_IN = 16000      # mic / VAD / Whisper rate
SAMPLE_RATE_TTS = 24000     # Kokoro output rate
PLAY_SECONDS = 0.1          # how long the fake output stream stays "active" after sd.play
PLAYBACK_TIMEOUT = 30       # seconds to wait for the playback worker to drain the queue
FRAME_SAMPLES = 480         # 30 ms at 16 kHz, matches mic_connect.FRAME_SIZE
USER_VOICE = "af_heart"     # a different voice from Beemo's, so the "user" is distinguishable in recordings


class AudioExhausted(Exception):
    """Raised by the fake mic once the scripted audio (plus trailing silence) has been fed."""


# Install stand-ins for hardware/network modules before any project module imports them.
def install_fakes():
    sd = types.ModuleType("sounddevice")
    sd._active_until = 0.0
    sd._on_play = lambda audio, rate: None   # the Sandbox points this at its current recording

    # Record the clip and stay "active" for a short fixed time, like a real stream the worker polls.
    def play(audio, samplerate=None, **k):
        sd._on_play(audio, samplerate)
        sd._active_until = time.monotonic() + PLAY_SECONDS

    def stop(*a, **k):
        sd._active_until = 0.0

    sd.play, sd.stop = play, stop
    sd.get_stream = lambda: types.SimpleNamespace(active=time.monotonic() < sd._active_until)

    def _no_mic(*a, **k):
        raise RuntimeError("sandbox: the real microphone must not be opened")
    sd.RawInputStream = _no_mic
    sys.modules["sounddevice"] = sd

    ww = types.ModuleType("wake_word_detect")
    ww.wake_word_detect = lambda stream: None   # behave as if "Beemo" was just heard
    sys.modules["wake_word_detect"] = ww


# Load KEY=VALUE lines into os.environ without overriding values already set (same rule as python-dotenv).
def load_env_file(path):
    if not path or not Path(path).exists():
        return
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


# Canned weather so tests do not need an API key; "atlantis" mimics an unknown location.
def fake_weather(location, detailed=False):
    if "atlantis" in location.lower():
        raise ValueError(f"Location not found: {location}")
    return ("light rain", 54.3, 51.0, 58.1, 49.7, 9.2, 0.4)


# Check the server answers a 1-token completion; returns an error string or None.
def llm_preflight(rr):
    import requests
    if not rr.HOST:
        return "LLM_HOST is not set (pass --env-file or export it)"
    try:
        res = requests.post(rr.HOST, json={
            "model": rr.MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1,
        }, timeout=(5, rr.READ_TIMEOUT))
    except requests.RequestException as e:
        return f"LLM server unreachable at {rr.HOST}: {e}"
    if res.status_code != 200:
        return f"LLM server returned HTTP {res.status_code}: {res.text[:200]}"
    return None


# Convert Kokoro float audio (tensor or array) to a flat float32 numpy array.
def to_np(audio):
    return np.asarray(audio.numpy() if hasattr(audio, "numpy") else audio, dtype=np.float32).reshape(-1)


# Write float32 audio in [-1, 1] as a 16-bit mono WAV.
def write_wav(path, audio, rate):
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


class Sandbox:
    """Owns the patched project modules and the per-scenario recordings."""

    # Import the project from `src` with fakes installed and patch the outward-facing calls.
    def __init__(self, src, args):
        self.args = args
        sys.path.insert(0, str(src))
        install_fakes()

        import req_to_res, smart_home, tts
        self.rr, self.tts = req_to_res, tts
        self.resolve_device = smart_home.resolve_device
        self.transcribe = self.vad = None
        self.reset()
        sys.modules["sounddevice"]._on_play = self._record_play   # dynamic, so reset() may rebind the lists

        rr = req_to_res
        rr.control_device = self._fake_control_device
        rr.playsound = lambda *a, **k: self.chimes.append(1)   # the "task complete" chime
        if not args.live_weather:
            rr.get_weather = fake_weather
        self._real_enqueue = tts.enqueue_tts
        rr.enqueue_tts = self._capture_tts
        real_failed = rr._llm_failed

        # Record LLM failures (they are otherwise only printed) before the normal handling runs.
        def failed(err):
            self.llm_errors.append(repr(err))
            real_failed(err)
        rr._llm_failed = failed

    # Clear conversation state and recordings so every scenario starts fresh.
    def reset(self):
        self.rr.context = []
        self.spoken, self.device_actions, self.llm_errors, self.tts_audio, self.chimes = [], [], [], [], []
        self.play_rates = []

    # Called by the fake sd.play: keep what the real playback worker would have sent to the speakers.
    def _record_play(self, audio, rate):
        self.tts_audio.append(to_np(audio))
        self.play_rates.append(rate)

    # Stand-in for smart_home.control_device: real name resolution, recorded action, no network.
    async def _fake_control_device(self, device, action):
        self.resolve_device(device)     # raises DeviceNotFoundError exactly like the real one
        self.device_actions.append({"device": device.lower(), "action": action})

    # Record what Beemo says; optionally synthesise it with the real Kokoro pipeline (the worker plays it).
    def _capture_tts(self, text):
        self.spoken.append(text)
        if not self.args.no_tts_synth:
            self._real_enqueue(text)

    # Wait until the playback worker has finished every queued clip; returns an error string or None.
    # Uses the queue's task counter, not is_speaking(), which is racy between queue.get and the flag being set.
    def wait_playback(self):
        q = self.tts._audio_queue
        deadline = time.monotonic() + PLAYBACK_TIMEOUT
        with q.all_tasks_done:
            while q.unfinished_tasks:
                if not self.tts._worker_thread.is_alive() or time.monotonic() > deadline:
                    break
                q.all_tasks_done.wait(0.2)
            if not q.unfinished_tasks:
                return None
        dead = not self.tts._worker_thread.is_alive()
        self.tts.stop_tts()   # drops leftovers (so they do not leak into the next scenario) and lets a live worker finish cleanly
        with q.all_tasks_done:   # a clip taken by a dead worker never got its task_done
            q.unfinished_tasks = 0
        self.tts.start_tts()
        return "playback worker died" if dead else "playback stalled"

    # Synthesise the user's line and run it through the real VAD endpointer and Whisper; returns the transcript.
    def hear(self, text, out_dir, sid):
        if self.vad is None:
            import vad, transcribe
            self.vad, self.transcribe = vad, transcribe
        speech = np.concatenate([to_np(a) for _, _, a in self.tts.pipeline(text, voice=USER_VOICE, speed=1.0)])
        n_out = int(len(speech) * SAMPLE_RATE_IN / SAMPLE_RATE_TTS)
        speech16 = np.interp(np.linspace(0, len(speech) - 1, n_out), np.arange(len(speech)), speech)
        audio = np.concatenate([np.zeros(SAMPLE_RATE_IN // 2), speech16, np.zeros(SAMPLE_RATE_IN * 2)])
        write_wav(out_dir / f"{sid}.user.wav", audio, SAMPLE_RATE_IN)
        pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()
        step = FRAME_SAMPLES * 2

        # Fake mic: yield 30 ms frames of the scripted audio, then stop the VAD loop instead of hanging.
        def frames():
            for i in range(0, len(pcm) - step + 1, step):
                yield pcm[i:i + step]
            raise AudioExhausted

        self.vad.stream_frames = frames
        try:
            utterance = next(self.vad.process())
        except AudioExhausted:   # the fake mic ran dry before the VAD found an utterance
            return None
        return self.transcribe.transcribe(utterance)

    # Run one scenario and return its result record (never raises).
    def run(self, sc, out_dir):
        self.reset()
        t0 = time.perf_counter()
        rec = {"id": sc["id"], "mode": sc.get("mode", "text"), "say": sc["say"], "errors": []}
        try:
            transcript = sc["say"]
            if rec["mode"] == "audio":
                transcript = self.hear(sc["say"], out_dir, sc["id"])
                rec["transcript"] = transcript
                if not transcript:
                    rec["errors"].append("VAD/Whisper produced no transcript")
                    return self._finish(rec, t0)
            rec["result"] = asyncio.run(self.rr.send_to_model(transcript))
        except Exception as e:
            rec["errors"].append(f"crashed: {e!r}")
        if not self.args.no_tts_synth:
            stalled = self.wait_playback()
            if stalled:
                rec["errors"].append(stalled)
            if any(r != SAMPLE_RATE_TTS for r in self.play_rates):
                rec["errors"].append(f"playback sample rates {sorted(set(self.play_rates))} != {SAMPLE_RATE_TTS}")
        rec["tool_calls"] = [
            {"name": c["function"]["name"], "args": c["function"]["arguments"]}
            for m in self.rr.context if m.get("tool_calls") for c in m["tool_calls"]
        ]
        if self.tts_audio:
            write_wav(out_dir / f"{sc['id']}.beemo.wav", np.concatenate(self.tts_audio), SAMPLE_RATE_TTS)
        rec["errors"] += check(sc.get("expect", {}), rec, self)
        return self._finish(rec, t0)

    # Attach recordings and timing to the result record.
    def _finish(self, rec, t0):
        rec.update(spoken=self.spoken, chime=bool(self.chimes), device_actions=self.device_actions, llm_errors=self.llm_errors,
                   tts_seconds=round(sum(len(a) for a in self.tts_audio) / SAMPLE_RATE_TTS, 2),
                   seconds=round(time.perf_counter() - t0, 2), passed=not rec["errors"])
        return rec


# Context must start at a user message and every tool call must have exactly one tool reply.
def context_problems(ctx):
    problems = []
    if ctx and ctx[0].get("role") != "user":
        problems.append("context does not start with a user message")
    for i, m in enumerate(ctx):
        if m.get("tool_calls"):
            want = [c["id"] for c in m["tool_calls"]]
            got = [r.get("tool_call_id") for r in ctx[i + 1:i + 1 + len(want)] if r.get("role") == "tool"]
            if got != want:
                problems.append(f"tool calls {want} not answered in order (got {got})")
    return problems


# Compare a scenario's recordings with its expectations; returns a list of failure messages.
def check(expect, rec, sb):
    errs = list(context_problems(sb.rr.context))
    if sb.llm_errors:
        errs.append(f"LLM error during turn: {sb.llm_errors}")
    if not "".join(sb.spoken).strip() and not sb.chimes:   # a successful device action answers with a chime only
        errs.append("Beemo gave no feedback (no speech, no chime)")
    if not sb.args.no_tts_synth and sb.spoken and not sb.tts_audio:
        errs.append("TTS produced no audio")

    names = [c["name"] for c in rec["tool_calls"]]
    if "tool" in expect:
        if expect["tool"] is None and names:
            errs.append(f"expected no tool call, got {names}")
        elif expect["tool"] and expect["tool"] not in names:
            errs.append(f"expected tool {expect['tool']}, got {names or 'none'}")
    if "device_actions" in expect and sb.device_actions != expect["device_actions"]:
        errs.append(f"device actions {sb.device_actions} != expected {expect['device_actions']}")
    if expect.get("location_contains"):
        locs = [json.loads(c["args"] or "{}").get("location", "") for c in rec["tool_calls"]
                if c["name"] == "get_weather"]
        if not any(expect["location_contains"].lower() in (l or "").lower() for l in locs):
            errs.append(f"weather location {locs} does not mention {expect['location_contains']!r}")
    for word in expect.get("transcript_contains", []):
        if word.lower() not in (rec.get("transcript") or "").lower():
            errs.append(f"transcript {rec.get('transcript')!r} is missing {word!r}")
    spoken = " ".join(sb.spoken).lower()
    if expect.get("spoken_any") and not any(w.lower() in spoken for w in expect["spoken_any"]):
        errs.append(f"reply mentions none of {expect['spoken_any']}")
    return errs


# Parse args, set up the sandbox, run the scenarios (retrying failures) and write report.json.
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="sandbox copy of the project to test")
    p.add_argument("--out", required=True, help="directory for report.json and WAV recordings")
    p.add_argument("--env-file", help="env file with LLM_HOST / LLM_MODEL (never overrides exported vars)")
    p.add_argument("--scenarios", default=str(HERE / "scenarios.json"))
    p.add_argument("--only", action="append", help="run only these scenario ids (repeatable)")
    p.add_argument("--skip-audio", action="store_true", help="skip scenarios that go through VAD + Whisper")
    p.add_argument("--no-tts-synth", action="store_true", help="record Beemo's text but skip Kokoro synthesis")
    p.add_argument("--live-weather", action="store_true", help="call the real weather API")
    p.add_argument("--retries", type=int, default=1, help="re-runs for a failing scenario (LLM is stochastic)")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    load_env_file(args.env_file)
    os.environ["DEVICES_FILE"] = str(HERE / "fixtures" / "devices.json")
    if not os.environ.get("CITY"):   # CI may export it empty
        os.environ["CITY"] = "Chicago"

    scenarios = json.loads(Path(args.scenarios).read_text())
    scenarios = [s for s in scenarios if (not args.only or s["id"] in args.only)
                 and not (args.skip_audio and s.get("mode") == "audio")]

    try:
        sb = Sandbox(Path(args.src).resolve(), args)
    except Exception as e:
        print(f"SANDBOX SETUP FAILED: importing the project crashed: {e!r}")
        return 2
    problem = llm_preflight(sb.rr)
    if problem:
        print(f"SANDBOX SETUP FAILED: {problem}")
        return 2
    if not args.no_tts_synth:
        sb.tts.start_tts()   # the real playback worker, fed by tts.enqueue_tts and draining into the fake sd.play

    results = []
    for sc in scenarios:
        print(f"\n===== scenario {sc['id']} ({sc.get('mode', 'text')}): {sc['say']!r}")
        rec = sb.run(sc, out)
        for attempt in range(args.retries):
            if rec["passed"]:
                break
            print(f"----- {sc['id']} failed ({rec['errors']}), retry {attempt + 1}")
            first = rec
            rec = sb.run(sc, out)
            rec["flaky"] = rec["passed"]
            rec["first_attempt_errors"] = first["errors"]
        results.append(rec)

    if not args.no_tts_synth:
        sb.tts.stop_tts()
    (out / "report.json").write_text(json.dumps(results, indent=2))
    print("\n===== SUMMARY")
    for r in results:
        status = "PASS" + (" (flaky)" if r.get("flaky") else "") if r["passed"] else "FAIL"
        print(f"{status:14} {r['id']:22} {r['seconds']:6.1f}s  tools={[c['name'] for c in r['tool_calls']]}")
        for e in r["errors"]:
            print(f"{'':14} - {e}")
    print(f"report: {out / 'report.json'}")
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
