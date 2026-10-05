"""Makes `src` importable when running pytest from the module root.

Also pins the working directory to the module root so relative paths such as
`models/labels.txt` resolve whether pytest is invoked from here or from the
parent project directory.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)