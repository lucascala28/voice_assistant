"""Pure, dependency-free splitter that turns a streaming LLM text buffer into speakable chunks."""
import re

TERMINATORS = ".?!"
CLOSERS = "\"')]}*_`”’"        # characters that may trail a sentence end
EARLY_MARKS = ",;:"            # extra cut points for the first chunk only
EARLY_MIN = 25                 # first chunk: don't cut early before this many chars
ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e",
    "approx", "inc", "ltd", "u.s", "u.k", "a.m", "p.m", "fig",
}
_WORD = re.compile(r"[A-Za-z.]*$")


# Return the index just past the first real sentence end in buf, or None if there is none yet.
def _sentence_end(buf, final):
    i, n = 0, len(buf)
    while i < n:
        if buf[i] not in TERMINATORS:
            i += 1
            continue
        j = i
        while j < n and buf[j] in TERMINATORS:      # "...", "?!"
            j += 1
        k = j
        while k < n and buf[k] in CLOSERS:           # '."', ".**", ".)"
            k += 1
        if k < n and not buf[k].isspace() and k == j:
            i = j                                    # "72.5", "e.g" : not a boundary
            continue
        if k == n and k == j and not final:
            return None                              # need one more char to tell "72." from "72.5"
        if buf[j - 1] == "." and j - i == 1 and _is_abbrev_or_list(buf, i):
            i = j
            continue
        return k
    return None


# True if the single period at index i belongs to an abbreviation or a "1." list marker.
def _is_abbrev_or_list(buf, i):
    word = _WORD.search(buf[:i]).group(0).lower().lstrip(".")   # keeps inner dots, e.g. "e.g"
    if word in ABBREVIATIONS:
        return True
    line_start = buf.rfind("\n", 0, i) + 1
    return buf[line_start:i].strip().isdigit()       # "1." at the start of a line


# Return the index just past an early cut point (first chunk only), or None.
def _early_end(buf):
    for i in range(EARLY_MIN, len(buf)):
        c = buf[i]
        if c == "\n":
            return i + 1
        if c in EARLY_MARKS and i + 1 < len(buf) and buf[i + 1].isspace():
            return i + 1
    return None


# Remove markdown that should not be read aloud and collapse whitespace.
def clean_for_speech(text):
    text = re.sub(r"(?m)^\s*[-+•]\s+", "", text)      # bullet markers
    text = re.sub(r"[*#`]+", "", text)
    return re.sub(r"\s+", " ", text).strip()


# Split buffer into (speakable chunk, remainder). chunk == "" means nothing is ready yet.
def next_chunk(buffer, first=False, final=False):
    end = _sentence_end(buffer, final)
    nl = buffer.find("\n")
    if nl != -1 and buffer[:nl].strip() and (end is None or nl + 1 < end):
        end = nl + 1                                 # a line break is a natural pause
    if first:
        early = _early_end(buffer)
        if early is not None and (end is None or early < end):
            end = early
    if end is None:
        if final:
            return clean_for_speech(buffer), ""
        return "", buffer
    return clean_for_speech(buffer[:end]), buffer[end:].lstrip()


class Chunker:
    """Stateful wrapper: feed() tokens as they stream, flush() at end of stream."""

    # Start with an empty buffer; the next chunk is the reply's first.
    def __init__(self):
        self.buf = ""
        self.first = True

    # Add streamed text; return the list of chunks that are now ready to speak.
    def feed(self, text):
        self.buf += text
        return self._drain(final=False)

    # Return everything still buffered (call at [DONE]).
    def flush(self):
        return self._drain(final=True)

    # Pull chunks out of the buffer until nothing more is ready.
    def _drain(self, final):
        out = []
        while self.buf.strip():
            chunk, rest = next_chunk(self.buf, self.first, final)
            if rest == self.buf and not chunk:
                break
            self.buf = rest
            if chunk:
                out.append(chunk)
                self.first = False
        if not self.buf.strip():
            self.buf = ""
        return out


class ThinkFilter:
    """Drops text between <think> and </think>; tags may be split across streamed tokens.
    A stray </think> with no opening tag drops the not-yet-emitted text before it (text streamed
    in earlier tokens has already been spoken and cannot be recalled)."""

    OPEN, CLOSE = "<think>", "</think>"

    # Start outside a think block with nothing held back.
    def __init__(self):
        self.inside = False
        self.held = ""

    # Return the visible part of `text`, holding back a trailing partial tag until the next token.
    def feed(self, text):
        buf, out = self.held + text, ""
        self.held = ""
        while buf:
            if self.inside:
                tags = (self.CLOSE,)
                i = buf.find(self.CLOSE)
                if i >= 0:
                    buf, self.inside = buf[i + len(self.CLOSE):], False
                    continue
            else:
                tags = (self.OPEN, self.CLOSE)
                i, j = buf.find(self.OPEN), buf.find(self.CLOSE)
                if j >= 0 and (i < 0 or j < i):          # closing tag with no opener: drop what came before
                    buf = buf[j + len(self.CLOSE):]
                    continue
                if i >= 0:
                    out, buf, self.inside = out + buf[:i], buf[i + len(self.OPEN):], True
                    continue
            keep = max((k for tag in tags for k in range(min(len(tag) - 1, len(buf)), 0, -1)
                        if tag.startswith(buf[-k:])), default=0)
            if not self.inside:
                out += buf[:len(buf) - keep]
            self.held = buf[len(buf) - keep:]
            break
        return out

    # Return any held-back text that turned out not to be a tag (call at end of stream).
    def flush(self):
        out = "" if self.inside else self.held
        self.held = ""
        return out
