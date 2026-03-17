import sounddevice as sd

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_MS / 1000)

def stream_frames():
    with sd.RawInputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16', blocksize=FRAME_SIZE) as stream:
        print("Mic connnected, listening...\n")
        while True:
            frame, _ = stream.read(FRAME_SIZE)
            yield bytes(frame)

