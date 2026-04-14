import sounddevice as sd
import numpy as np
from kokoro import KPipeline

pipeline = KPipeline(lang_code="a")

def speak(text):
    generator = pipeline(text, voice="bm_lewis", speed=1.0)
    for _, _, audio in generator:
        sd.play(audio, samplerate=24000)
        sd.wait()