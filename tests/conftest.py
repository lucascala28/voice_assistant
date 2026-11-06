import sys
from pathlib import Path

# Make the flat repo modules (text_chunker, sse_parse, latency, ...) importable from tests/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
