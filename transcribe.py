import numpy as np
from vad import process
from faster_whisper import WhisperModel

MODEL_SIZE = "tiny"  # tiny, base, small
SAMPLE_RATE = 16000

model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

def transcribe(speech_bytes):
    audio = np.frombuffer(speech_bytes, dtype=np.int16).astype(np.float32) / 32678.0
    segments, _ = model.transcribe(audio, beam_size=1)
    transcript = " ".join(segment.text for segment in segments).strip()

    return transcript

def collect_and_transcribe():
    for speech_bytes in process():
        transcript = transcribe(speech_bytes)
        if transcript:
            yield transcript

collect_and_transcribe()