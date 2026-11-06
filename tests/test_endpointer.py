from endpointer import Endpointer


# Push a sequence of booleans (True = speech) and return the list of non-empty events with their frame index.
def drive(ep, pattern):
    events = []
    for i, is_speech in enumerate(pattern):
        event, audio = ep.push(b"f", is_speech)
        if event:
            events.append((i, event, audio))
    return events


def test_normal_utterance():
    ep = Endpointer(silence_frames=3, min_speech_frames=2, idle_frames=100)
    events = drive(ep, [False, True, True, True, False, False, False])
    assert [(i, e) for i, e, _ in events] == [(6, "utterance")]
    assert events[0][2] == b"f" * 6          # speech + trailing silence, nothing from before speech started


def test_blip_then_real_speech_is_kept():
    ep = Endpointer(silence_frames=3, min_speech_frames=3, idle_frames=100)
    pattern = [True] + [False] * 3 + [False] * 5 + [True] * 4 + [False] * 3     # tail of wake word, pause, command
    events = drive(ep, pattern)
    assert [e for _, e, _ in events] == ["utterance"]
    assert events[0][2] == b"f" * 7          # only the command; the blip was discarded


def test_min_speech_guard_without_followup_times_out():
    ep = Endpointer(silence_frames=2, min_speech_frames=3, idle_frames=10)
    events = drive(ep, [True, False, False] + [False] * 20)
    assert [(i, e) for i, e, _ in events][0] == (9, "timeout")      # total wait stays bounded by idle_frames


def test_idle_timeout_without_speech():
    ep = Endpointer(silence_frames=2, min_speech_frames=2, idle_frames=5)
    assert [(i, e) for i, e, _ in drive(ep, [False] * 8)][0] == (4, "timeout")


def test_no_timeout_while_speaking():
    ep = Endpointer(silence_frames=2, min_speech_frames=2, idle_frames=5)
    events = drive(ep, [True] * 10 + [False, False])
    assert [e for _, e, _ in events] == ["utterance"]
