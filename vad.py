from mic_connect import stream_frames, SAMPLE_RATE, FRAME_MS
from wake_word_detect import wake_word_detect
import os
import time
import webrtcvad
import latency
from endpointer import Endpointer

VAD_AGGRO=2
# Trailing silence (in FRAME_MS frames) that ends an utterance; 18 x 30ms = ~540ms. Override with VAD_SILENCE_FRAMES.
SILENCE_THRESHOLD=int(os.getenv("VAD_SILENCE_FRAMES", "18"))
# Speech frames required before silence may end the utterance (filters clicks/noise blips).
MIN_SPEECH_FRAMES=int(os.getenv("VAD_MIN_SPEECH_FRAMES", "6"))
# Seconds to wait for speech after the wake word before going back to wake-word listening.
IDLE_FRAMES=int(float(os.getenv("VAD_IDLE_TIMEOUT_S", "6")) * 1000 / FRAME_MS)

vad = webrtcvad.Vad(VAD_AGGRO)

# Wait for the wake word, then yield one utterance (raw PCM bytes) each time trailing silence ends it.
def process():
    stream = stream_frames()
    while True:
        print("Detecting wake word ...")
        wake_word_detect(stream)

        ep = Endpointer(SILENCE_THRESHOLD, MIN_SPEECH_FRAMES, IDLE_FRAMES)
        for frame in stream:
            event, audio = ep.push(frame, vad.is_speech(frame, SAMPLE_RATE))
            if event == "utterance":
                t_endpoint = time.perf_counter()
                latency.new_turn()
                latency.mark("endpoint", t_endpoint)
                latency.mark("speech_end", t_endpoint - SILENCE_THRESHOLD * FRAME_MS / 1000)
                print("\nSilence threshold reached. Sending data to transcription model...")
                yield audio
                break
            if event == "timeout":
                print("\nNo speech after wake word, going back to sleep.")
                break
