"""Shared plumbing for the experiment scripts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results"


def results_path(name: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / name


def banner(text: str) -> None:
    print()
    print("=" * 72)
    print(text)
    print("=" * 72)
