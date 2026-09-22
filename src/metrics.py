"""Accuracy measures for an approximate PageRank vector.

The base paper monitors a single quantity, the eigen-residual
``||A x - nu x||_2``.  That is the right stopping criterion for an eigenvalue
solver, but it is not what a ranking application cares about.  A search engine
needs the *order* of the top results to be correct; it is indifferent to the
fourth significant figure of each score, and indifferent entirely to the tail.

This module therefore provides the ranking-aware measures used in Phase 5:

``l1_error``
    Total-variation-style distance between two rank vectors.
``kendall_tau`` / ``spearman``
    Order agreement, computed over the union of the two top-``k`` sets so that
    the statistic reflects the part of the ranking anyone looks at.
``precision_at_k``
    Fraction of the true top-``k`` recovered, ignoring order entirely.

Tracking these per iteration answers the question the residual cannot: how many
matrix-vector products until the *ranking* is right?
"""

from __future__ import annotations

import numpy as np

from .google_matrix import normalise_ranking

__all__ = [
    "l1_error",
    "linf_error",
    "precision_at_k",
    "kendall_tau",
    "spearman",
    "rank_metrics",
    "iterations_to_threshold",
]


def l1_error(x: np.ndarray, reference: np.ndarray) -> float:
    """``||x - reference||_1`` after both are normalised to unit 1-norm."""
    return float(np.abs(normalise_ranking(x) - normalise_ranking(reference)).sum())


def linf_error(x: np.ndarray, reference: np.ndarray) -> float:
    return float(np.abs(normalise_ranking(x) - normalise_ranking(reference)).max())


def precision_at_k(x: np.ndarray, reference: np.ndarray, k: int = 100) -> float:
    """Overlap between the top-``k`` of ``x`` and the top-``k`` of ``reference``."""
    k = min(k, x.size)
    top_x = np.argpartition(-normalise_ranking(x), k - 1)[:k]
    top_ref = np.argpartition(-normalise_ranking(reference), k - 1)[:k]
    return len(np.intersect1d(top_x, top_ref)) / k


def _top_union(x: np.ndarray, reference: np.ndarray, k: int) -> np.ndarray:
    k = min(k, x.size)
    top_x = np.argpartition(-x, k - 1)[:k]
    top_ref = np.argpartition(-reference, k - 1)[:k]
    return np.union1d(top_x, top_ref)


def kendall_tau(x: np.ndarray, reference: np.ndarray, k: int = 100) -> float:
    """Kendall's tau-b over the union of the two top-``k`` sets."""
    from scipy.stats import kendalltau

    xs, rs = normalise_ranking(x), normalise_ranking(reference)
    idx = _top_union(xs, rs, k)
    if idx.size < 2:
        return float("nan")
    return float(kendalltau(xs[idx], rs[idx]).statistic)


def spearman(x: np.ndarray, reference: np.ndarray, k: int = 100) -> float:
    """Spearman rank correlation over the union of the two top-``k`` sets."""
    from scipy.stats import spearmanr

    xs, rs = normalise_ranking(x), normalise_ranking(reference)
    idx = _top_union(xs, rs, k)
    if idx.size < 2:
        return float("nan")
    return float(spearmanr(xs[idx], rs[idx]).statistic)


def rank_metrics(
    x: np.ndarray, reference: np.ndarray, ks: tuple[int, ...] = (10, 100, 1000)
) -> dict[str, float]:
    """All ranking measures at once, for one iterate."""
    out: dict[str, float] = {
        "l1_error": l1_error(x, reference),
        "linf_error": linf_error(x, reference),
    }
    for k in ks:
        if k > x.size:
            continue
        out[f"precision@{k}"] = precision_at_k(x, reference, k)
        out[f"kendall@{k}"] = kendall_tau(x, reference, k)
    return out


def iterations_to_threshold(values, threshold: float, rising: bool = True) -> int | None:
    """Index of the first entry that crosses ``threshold``.

    With ``rising=True`` the first value ``>= threshold`` is reported (for
    precision and correlation, which increase towards one); with
    ``rising=False`` the first value ``<= threshold`` (for error norms).
    Returns ``None`` if the threshold is never crossed.
    """
    for i, v in enumerate(values):
        if np.isnan(v):
            continue
        if (rising and v >= threshold) or (not rising and v <= threshold):
            return i
    return None
