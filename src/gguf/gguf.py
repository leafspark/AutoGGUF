import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Compatibility for people trying to import gguf/gguf.py directly instead of as a package.
importlib.invalidate_caches()
import gguf  # noqa: E402

importlib.reload(gguf)
