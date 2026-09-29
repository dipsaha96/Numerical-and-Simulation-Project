"""Shared plumbing for the ca-HepPh experiment suite.

This suite is deliberately narrow.  It runs the three algorithms of the base
paper, unmodified, on a single PageRank problem: the undirected co-authorship
graph ``ca-HepPh``.  Nothing here is safeguarded, patched or tuned -- every
result is the published method doing exactly what the paper says it will do.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIGURE_DIR = ROOT / "figure"
RESULT_DIR = ROOT / "result"

# The one graph this suite uses for PageRank.
GRAPH = "ca-HepPh"

# The damping factors swept throughout.  0.85 is the classic choice; the larger
# values are where the spectral gap closes and acceleration matters most.
DAMPINGS = [0.85, 0.90, 0.95, 0.99, 0.999]

TOL = 1e-12
MAX_ITER = 200_000


def result_path(name: str) -> Path:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    return RESULT_DIR / name


def banner(text: str) -> None:
    print()
    print("=" * 74)
    print(text)
    print("=" * 74)


def save_figure(fig, name: str) -> None:
    """Write a figure into ``figure/`` as both PDF (LaTeX) and PNG (slides)."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURE_DIR / f"{name}.{ext}", bbox_inches="tight",
                    dpi=200 if ext == "png" else None)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"  wrote figure/{name}.pdf and .png")
