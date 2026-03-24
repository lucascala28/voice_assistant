import openwakeword
import numpy as np
from openwakeword import Model

openwakeword.utils.download_models()
oww = Model(wakeword_models=["hey_mycroft"], inference_framework="onnx")

def wake_word_detect(stream):
    for frame in stream:
        audio = np.frombuffer(frame, dtype=np.int16)
        prediction = oww.predict(audio)
        if prediction["hey_mycroft"] > 0.8:
            print("Wake word detected!")
            oww.reset()
            return