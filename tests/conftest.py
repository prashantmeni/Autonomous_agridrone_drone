import sys
from pathlib import Path

# tests/conftest.py -> parents[1] is the repo root. Both entries are needed:
# src for the `drone` package, and the root for `ai_disease_detection`.
ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT):
    entry = str(path)
    if path.is_dir() and entry not in sys.path:
        sys.path.insert(0, entry)
