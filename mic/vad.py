from mic_connect import stream_frames, SAMPLE_RATE, FRAME_MS
import webrtcvad

VAD_AGGRO=2
SILENCE_THRESHOLD=2

vad = webrtcvad.Vad(VAD_AGGRO)

def process():
    buffer = []
    silence = 0
    still_speaking = False
    for frame in stream_frames():
        is_speech = vad.is_speech(frame, SAMPLE_RATE)

        if is_speech:
            silence = 0
            still_speaking = True
            buffer.append(frame)
        
        elif still_speaking:
            silence += 1
            buffer.append(frame)

            if silence >= SILENCE_THRESHOLD:
                print("Silence threshold reached. Sending data to transcription model...")
                yield b"".join(buffer)
                buffer = []
                silence = 0
                still_speaking = False

process()

