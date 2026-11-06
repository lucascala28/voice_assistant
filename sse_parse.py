"""Pure, dependency-free helpers for OpenAI-style (LM Studio) streaming responses."""
import json


# Parse one raw SSE line into (kind, payload): kind is "skip", "done", "error" or "delta".
def parse_line(line):
    if isinstance(line, bytes):
        line = line.decode("utf-8", errors="replace")
    line = line.strip()
    if not line or line.startswith(":"):
        return "skip", None
    if line.startswith("data:"):
        line = line[5:].strip()
    if line == "[DONE]":
        return "done", None
    try:
        chunk = json.loads(line)
    except json.JSONDecodeError:
        return "skip", None
    if not isinstance(chunk, dict):
        return "skip", None
    if chunk.get("error"):
        return "error", chunk["error"]
    choices = chunk.get("choices") or []
    if not choices:
        return "skip", None                       # e.g. a trailing usage-only chunk
    delta = dict(choices[0].get("delta") or {})
    if choices[0].get("finish_reason"):
        delta["finish_reason"] = choices[0]["finish_reason"]   # surfaced so callers can warn on "length"
    return "delta", delta


# Spoken text in a delta ("" when content is null/missing, e.g. role-only or tool-call chunks).
def content_of(delta):
    return delta.get("content") or ""


class ToolCallAccumulator:
    """Collects streamed tool_call fragments (keyed by their index) across chunks."""

    # Start with no accumulated calls.
    def __init__(self):
        self.calls = {}

    # Merge the tool_calls list from one delta into the accumulated calls.
    def add(self, delta):
        for tc in delta.get("tool_calls") or []:
            slot = self.calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "args": ""})
            slot["id"] = tc.get("id") or slot["id"]
            fn = tc.get("function") or {}
            slot["name"] = fn.get("name") or slot["name"]
            slot["args"] += fn.get("arguments") or ""

    # All accumulated calls in index order; a missing server id becomes "call_<index>".
    def all(self):
        return [{**self.calls[i], "id": self.calls[i]["id"] or f"call_{i}"} for i in sorted(self.calls)]
