"""Benchmark: Kokoro synthesis time on cpu vs mps. Run: uv run python scripts/bench_tts.py
Benchmark only; does not change the device tts.py uses."""
import time
from kokoro import KPipeline

TEXT = "It is currently seventy two degrees and sunny in Boston, with a light breeze from the west."
RUNS = 5


# Time RUNS syntheses of TEXT on one device (first run reported separately as cold start).
def bench(device):
    try:
        pipe = KPipeline(lang_code="a", device=device)
        pipe.load_voice("bm_lewis")
    except Exception as e:
        print(f"{device}: unavailable ({e})")
        return
    times = []
    for _ in range(RUNS):
        t = time.perf_counter()
        for _ in pipe(TEXT, voice="bm_lewis", speed=1.0):
            pass
        times.append(time.perf_counter() - t)
    print(f"{device}: cold {times[0]*1000:.0f}ms, warm avg {sum(times[1:])/(RUNS-1)*1000:.0f}ms")


if __name__ == "__main__":
    for dev in ("cpu", "mps"):
        bench(dev)
