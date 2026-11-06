"""Pure, dependency-free endpointing state machine (frame in -> event out) used by vad.py."""


class Endpointer:
    """Collects speech after a wake word and reports when the utterance ended or the wait timed out."""

    # silence_frames: trailing silence that ends an utterance; min_speech_frames: speech needed before that
    # counts; idle_frames: total frames since the wake word after which we give up if nobody has spoken.
    def __init__(self, silence_frames, min_speech_frames, idle_frames):
        self.silence_frames = silence_frames
        self.min_speech_frames = min_speech_frames
        self.idle_frames = idle_frames
        self.total = 0          # every frame since the wake word (bounds the total wait)
        self.buffer = []
        self.silence = 0
        self.speech = 0

    # Feed one frame; returns (event, audio) with event None, "utterance" (audio = joined frames) or "timeout".
    def push(self, frame, is_speech):
        self.total += 1
        if is_speech:
            self.silence = 0
            self.speech += 1
            self.buffer.append(frame)
        elif self.speech:
            self.silence += 1
            self.buffer.append(frame)
            if self.silence >= self.silence_frames:
                if self.speech >= self.min_speech_frames:
                    return "utterance", b"".join(self.buffer)
                # a blip (click, tail of the wake word): drop it and keep waiting in the same phase
                self.buffer, self.silence, self.speech = [], 0, 0
        if not self.speech and self.total >= self.idle_frames:
            return "timeout", None
        return None, None
