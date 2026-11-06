import sounddevice as sd
import numpy as np
from kokoro import KPipeline
import queue
import threading
import latency

VOICE = "bm_lewis"
pipeline = KPipeline(lang_code="a")
_audio_queue = queue.Queue()
_stop_event = threading.Event()
_worker_thread = None
_is_speaking = False

def playback_worker():
    global _is_speaking
    while not _stop_event.is_set():
        try:
            audio = _audio_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        
        _is_speaking = True
        sd.play(audio, samplerate=24000)
        latency.mark("audio_playing")
        while sd.get_stream().active:
            if _stop_event.is_set():
                sd.stop()
                break
            
            threading.Event().wait(0.05)
        
        _is_speaking = False
        _audio_queue.task_done()

def start_tts():
    global _worker_thread
    _stop_event.clear()
    _worker_thread = threading.Thread(target=playback_worker, daemon=True)
    _worker_thread.start()

def stop_tts():
    _stop_event.set()
    sd.stop()
    while not _audio_queue.empty():
        try:
            _audio_queue.get_nowait()
            _audio_queue.task_done()
        except queue.Empty:
            break
    if _worker_thread:
        _worker_thread.join(timeout=1)

# Load the voice and synthesize one short phrase (audio discarded) so the first real reply is not cold.
def warmup():
    pipeline.load_voice(VOICE)
    for _ in pipeline("Ready.", voice=VOICE, speed=1.0):
        pass

# Synthesize text with Kokoro (blocking) and queue the audio for the playback thread.
def enqueue_tts(text):
    generator = pipeline(text, voice=VOICE, speed=1.0)
    for _, _, audio in generator:
        if _stop_event.is_set():
            return
        latency.mark("first_tts_audio_ready")
        _audio_queue.put(audio)
    
def is_speaking():
    return _is_speaking or not _audio_queue.empty()