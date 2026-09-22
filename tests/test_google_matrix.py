"""Tests for the Google matrix operator -- the part most likely to be subtly wrong."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets import synthetic_web_graph                       # noqa: E402
from src.google_matrix import (                                    # noqa: E402
    build_google_matrix,
    edges_to_sparse,
    normalise_ranking,
)
from src.iterations import power_iteration                         # noqa: E402
from src.metrics import kendall_tau, l1_error, precision_at_k      # noqa: E402


@pytest.fixture(scope="module")
def graph():
    return synthetic_web_graph(n=2000, seed=3)


def test_link_matrix_is_column_stochastic(graph):
    P, dangling, _ = edges_to_sparse(*graph)
    sums = np.asarray(P.sum(axis=0)).ravel()
    assert np.allclose(sums[~dangling], 1.0)
    assert np.allclose(sums[dangling], 0.0)


def test_google_matrix_preserves_total_mass(graph):
    """``e^T G x = e^T x`` for *any* ``x``, not just probability vectors.

    This is the invariant the common ``||x||_1 = 1`` shortcut quietly assumes.
    The momentum iterates are 2-normalised and may be negative, so the operator
    has to hold for arbitrary input or the whole comparison is invalid.
    """
    G, _ = build_google_matrix(*graph, d=0.85)
    rng = np.random.default_rng(0)
    for x in (np.ones(G.n), rng.standard_normal(G.n), -rng.random(G.n),
              np.zeros(G.n)):
        assert G.matvec(x).sum() == pytest.approx(x.sum(), abs=1e-9)


def test_pagerank_is_the_fixed_point(graph):
    G, _ = build_google_matrix(*graph, d=0.85)
    x = normalise_ranking(power_iteration(G, tol=1e-14, max_iter=10_000).x)
    assert np.all(x > 0)                       # Perron-Frobenius
    assert x.sum() == pytest.approx(1.0)
    assert np.abs(G.matvec(x) - x).sum() < 1e-12


def test_dominant_eigenvalue_is_one(graph):
    G, _ = build_google_matrix(*graph, d=0.85)
    assert power_iteration(G, tol=1e-13, max_iter=10_000).nu == pytest.approx(1.0, abs=1e-9)


def test_matvec_handles_complex_input(graph):
    """ARPACK probes the operator with complex vectors when asked for complex
    eigenpairs; silently casting to float corrupts the spectral diagnostics."""
    G, _ = build_google_matrix(*graph, d=0.85)
    z = np.ones(G.n) + 1j * np.linspace(-1, 1, G.n)
    y = G.matvec(z)
    assert np.iscomplexobj(y)
    assert np.allclose(y.real, G.matvec(z.real))
    assert np.allclose(y.imag, G.matvec(z.imag))


def test_with_damping_reuses_structure(graph):
    G, _ = build_google_matrix(*graph, d=0.85)
    H = G.with_damping(0.99)
    assert H.P is G.P and H.d == 0.99 and G.d == 0.85


def test_self_loops_and_duplicates_are_dropped():
    src = np.array([0, 0, 1, 1, 2])
    dst = np.array([1, 1, 0, 1, 2])          # one duplicate, two self-loops
    P, dangling, ids = edges_to_sparse(src, dst)
    assert P.nnz == 2
    assert dangling[2]                        # node 2 had only a self-loop
    assert list(ids) == [0, 1, 2]


def test_normalise_ranking_fixes_sign():
    x = -np.array([0.2, 0.3, 0.5])
    out = normalise_ranking(x)
    assert np.all(out > 0) and out.sum() == pytest.approx(1.0)


def test_ranking_metrics_are_exact_on_identical_input(graph):
    G, _ = build_google_matrix(*graph, d=0.85)
    x = power_iteration(G, tol=1e-14, max_iter=10_000).x
    assert l1_error(x, x) == pytest.approx(0.0, abs=1e-15)
    assert precision_at_k(x, x, 100) == 1.0
    assert kendall_tau(x, x, 100) == pytest.approx(1.0)
