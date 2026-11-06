import os
import numpy as np
from vad import process
import latency
from faster_whisper import WhisperModel

MODEL_SIZE = os.getenv("WHISPER_MODEL", "tiny")  # tiny, tiny.en (English-only, downloads on first use), base, small
SAMPLE_RATE = 16000
LANGUAGE = "en"

model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

# Transcribe raw 16-bit PCM bytes to text (language fixed to English to skip detection).
def transcribe(speech_bytes):
    audio = np.frombuffer(speech_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    segments, _ = model.transcribe(audio, beam_size=1, language=LANGUAGE)
    transcript = " ".join(segment.text for segment in segments).strip()
    latency.mark("transcribe_done")

    return transcript

# Run one throwaway transcription of 1s of silence so the first real request is not cold.
def warmup():
    segments, _ = model.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), beam_size=1, language=LANGUAGE)
    list(segments)  # the segment generator is lazy; consume it so the work actually runs

# Yield a transcript for every utterance that the VAD captures (empty transcripts are dropped).
def collect_and_transcribe():
    for speech_bytes in process():
        transcript = transcribe(speech_bytes)
        if transcript:
            yield transcript
