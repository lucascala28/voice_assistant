from pathlib import Path

import numpy as np
import openwakeword
from openwakeword import Model
from openwakeword.utils import download_file

# openWakeWord's shared feature models aren't shipped with the pip package, fetch them if missing
for feature_model in openwakeword.FEATURE_MODELS.values():
    onnx_path = Path(feature_model["model_path"]).with_suffix(".onnx")
    if not onnx_path.exists():
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        download_file(feature_model["download_url"].replace(".tflite", ".onnx"), str(onnx_path.parent))

WWM_PATH = str(Path(__file__).parent / "data" / "beemo.onnx")
oww = Model(wakeword_models=[WWM_PATH], inference_framework="onnx")

def wake_word_detect(stream):
    for frame in stream:
        audio = np.frombuffer(frame, dtype=np.int16)
        prediction = oww.predict(audio)
        if prediction["beemo"] > 0.8:
            print("Wake word detected!")
            oww.reset()
            return