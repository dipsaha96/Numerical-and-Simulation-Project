"""One figure style for the whole suite, so every plot matches."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito, colour-blind safe.
STYLE = {
    "power":   dict(color="#0072B2", marker="o", label="Power (Alg. 1.1)"),
    "static":  dict(color="#D55E00", marker="s", label=r"Static $\beta=d^2/4$ (Alg. 1.2)"),
    "dynamic": dict(color="#009E73", marker="D", label=r"Dynamic $\beta_k$ (Alg. 3.1)"),
    "theory":  dict(color="#333333", marker="",  label="Theory"),
}


def use_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": ":",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "lines.linewidth": 1.7,
        "lines.markersize": 4.5,
    })


def markevery(n: int, target: int = 12) -> int:
    return max(1, n // target)
