"""Shared Matplotlib styling so every figure in the report matches.

Import :func:`use_style` once at the top of an experiment script and use
:data:`METHOD_STYLE` for per-method colours and markers.  Figures are written
as both PDF (for LaTeX) and PNG (for the slides).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

__all__ = ["use_style", "METHOD_STYLE", "save_figure", "FIGURE_DIR"]

FIGURE_DIR = Path(__file__).resolve().parent.parent / "figures"

# Colour-blind-safe palette (Okabe-Ito), consistent across every figure.
METHOD_STYLE = {
    "power":        dict(color="#0072B2", marker="o", label="Power (Alg. 1.1)"),
    "static":       dict(color="#D55E00", marker="s", label=r"Static $\beta=\lambda_2^2/4$ (Alg. 1.2)"),
    "static_d":     dict(color="#D55E00", marker="s", label=r"Static $\beta=d^2/4$ (Alg. 1.2)"),
    "dynamic_safe": dict(color="#009E73", marker="D", label="Dynamic + safeguards"),
    "dynamic_guard":dict(color="#56B4E9", marker="*", label="Dynamic + safeguards + rate guard"),
    "static_lo":    dict(color="#E69F00", marker="v", label=r"Static $0.99\,\beta_{opt}$"),
    "static_hi":    dict(color="#CC79A7", marker="^", label=r"Static $1.01\,\beta_{opt}$"),
    "dynamic":      dict(color="#009E73", marker="D", label="Dynamic $\\beta_k$ (Alg. 3.1)"),
    "dynamic_paper":dict(color="#CC79A7", marker="x", label="Dynamic, Alg. 3.1 verbatim"),
    "theory":       dict(color="#555555", marker="", label="Theory"),
}


def use_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": ":",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "lines.linewidth": 1.6,
        "lines.markersize": 4,
        "figure.autolayout": False,
    })


def save_figure(fig, name: str) -> None:
    """Write ``name`` into ``figures/`` as both PDF and PNG."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURE_DIR / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf")


def markevery(n: int, target: int = 12) -> int:
    """Marker stride that puts roughly ``target`` markers on a curve."""
    return max(1, n // target)
