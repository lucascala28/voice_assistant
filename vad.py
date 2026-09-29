from mic_connect import stream_frames, SAMPLE_RATE, FRAME_MS
from wake_word_detect import wake_word_detect
import webrtcvad

VAD_AGGRO=2
SILENCE_THRESHOLD=30

vad = webrtcvad.Vad(VAD_AGGRO)

def process():
    stream = stream_frames()
    while True:
        print("Detecting wake word ...")
        wake_word_detect(stream)

        buffer = []
        silence = 0
        still_speaking = False
        for frame in stream: 
            is_speech = vad.is_speech(frame, SAMPLE_RATE)

            if is_speech:
                silence = 0
                still_speaking = True
                buffer.append(frame)
            
            elif still_speaking:
                silence += 1
                buffer.append(frame)

                if silence >= SILENCE_THRESHOLD:
                    print("\nSilence threshold reached. Sending data to transcription model...")
                    yield b"".join(buffer)
                    buffer = []
                    silence = 0
                    still_speaking = False
                    break

process()

