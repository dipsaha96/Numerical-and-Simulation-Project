"""Correctness tests for the three iterations, independent of any dataset."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.iterations import (                                      # noqa: E402
    asymptotic_rate,
    dynamic_momentum,
    optimal_beta,
    power_iteration,
    predicted_speedup,
    static_momentum,
)


def diag(n=200):
    vals = np.arange(n, 0, -1, dtype=float)
    return sp.diags(vals).tocsr(), vals


def test_power_finds_dominant_eigenpair():
    A, vals = diag(100)
    res = power_iteration(A, tol=1e-12, max_iter=50_000)
    assert res.converged
    assert res.nu == pytest.approx(vals[0], rel=1e-10)
    assert abs(res.x[0]) == pytest.approx(1.0, abs=1e-8)     # e_1 is the eigenvector


def test_momentum_methods_agree_with_power():
    A, vals = diag(100)
    ref = power_iteration(A, tol=1e-12, max_iter=50_000)
    for res in (static_momentum(A, beta=optimal_beta(vals[1]), tol=1e-12, max_iter=50_000),
                dynamic_momentum(A, tol=1e-12, max_iter=50_000)):
        assert res.converged
        assert res.nu == pytest.approx(ref.nu, rel=1e-9)
        # eigenvectors agree up to sign
        assert min(np.linalg.norm(res.x - ref.x), np.linalg.norm(res.x + ref.x)) < 1e-6


def test_one_matvec_per_iteration():
    """Every method must cost one matrix-vector product per iteration.

    This is the premise of the whole comparison: reporting iteration counts is
    only fair if the per-iteration cost is identical.
    """
    A, vals = diag(80)
    for res in (power_iteration(A, tol=1e-11, max_iter=50_000),
                static_momentum(A, beta=optimal_beta(vals[1]), tol=1e-11, max_iter=50_000),
                dynamic_momentum(A, tol=1e-11, max_iter=50_000)):
        overhead = res.matvecs - res.n_iter
        assert 0 <= overhead <= 3, f"{res.method}: {overhead} extra matvecs"


def test_momentum_accelerates_symmetric_problem():
    """Theorem 3.1: acceleration is guaranteed in the symmetric case."""
    A, vals = diag(500)                     # r = 0.998
    p = power_iteration(A, tol=1e-12, max_iter=200_000)
    s = static_momentum(A, beta=optimal_beta(vals[1]), tol=1e-12, max_iter=200_000)
    dyn = dynamic_momentum(A, tol=1e-12, max_iter=200_000)
    assert s.converged and dyn.converged
    assert p.matvecs > 5 * s.matvecs
    assert p.matvecs > 5 * dyn.matvecs


def test_dynamic_matches_predicted_rate():
    """The measured speedup must track log(rho(r))/log(r)."""
    A, vals = diag(500)
    r = vals[1] / vals[0]
    p = power_iteration(A, tol=1e-12, max_iter=200_000)
    dyn = dynamic_momentum(A, tol=1e-12, max_iter=200_000)
    measured = p.matvecs / dyn.matvecs
    assert measured == pytest.approx(predicted_speedup(r), rel=0.35)


def test_dynamic_beta_converges_to_optimal():
    """beta_k must find lambda_2^2/4 without being told lambda_2."""
    A, vals = diag(500)
    dyn = dynamic_momentum(A, tol=1e-12, max_iter=200_000)
    assert np.median(dyn.beta[-50:]) == pytest.approx(optimal_beta(vals[1]), rel=0.05)


def test_lambda1_cap_does_not_clip_the_optimum():
    """Regression: the cap must not fire on a well-posed symmetric problem.

    Capping against ``nu_k`` instead of a known ``lambda_1`` silently clipped
    ``beta_k`` below ``lambda_2^2/4`` whenever ``r^2`` exceeded the slack factor,
    which quadrupled the iteration count on exactly the near-degenerate problems
    the method is for.
    """
    A, vals = diag(500)
    dyn = dynamic_momentum(A, tol=1e-12, max_iter=200_000, lambda1=float(vals[0]))
    assert dyn.safeguard_hits == 0
    assert dyn.converged


def test_rate_formulas():
    assert asymptotic_rate(0.0) == pytest.approx(0.0)
    assert asymptotic_rate(1.0) == pytest.approx(1.0)
    assert asymptotic_rate(0.85) == pytest.approx(0.5567262, abs=1e-6)
    assert predicted_speedup(0.85) == pytest.approx(3.6038, abs=1e-3)
    assert predicted_speedup(0.999) == pytest.approx(44.72, abs=0.05)
    # rho(r) < r strictly on (0, 1): momentum always helps a real spectrum
    for r in (0.1, 0.5, 0.9, 0.99):
        assert asymptotic_rate(r) < r


def test_nonsymmetric_needs_safeguards():
    """A non-normal matrix with complex eigenvalues breaks Algorithm 3.1.

    A scaled rotation block has purely imaginary eigenvalues, the worst case for
    the augmented matrix: with ``beta`` near ``lambda_1^2/4`` these produce
    ``|mu| = (|y| + sqrt(y^2 + 4 beta))/2 > sqrt(beta)``, overtaking the dominant
    mode.  This is the mechanism that defeats the method on the Google matrix,
    reproduced in eight dimensions.
    """
    n = 8
    A = np.zeros((n, n))
    A[0, 0] = 1.0
    for i in range(1, n - 1, 2):            # rotation blocks: eigenvalues +-0.9i
        A[i, i + 1], A[i + 1, i] = -0.9, 0.9
    naive = dynamic_momentum(A, tol=1e-12, max_iter=3000)
    safe = dynamic_momentum(A, tol=1e-12, max_iter=3000, lambda1=1.0, r_max=0.9,
                            backtrack_window=8, max_backtracks=20)
    assert not naive.converged, "expected Algorithm 3.1 to fail on a complex spectrum"
    assert safe.converged, "safeguarded variant must recover"
    assert safe.backtracks > 0


def test_backtracking_terminates_without_momentum():
    """Regression: with the ceiling at zero, backtracking must switch off.

    Otherwise the method re-primes from the same best iterate forever.  An
    unbudgeted version burned 7240 backtracks on cit-HepPh at d = 0.999 and
    never converged.
    """
    A, vals = diag(300)
    res = dynamic_momentum(A, tol=1e-12, max_iter=60_000, lambda1=float(vals[0]),
                           r_max=0.997, backtrack_window=5, max_backtracks=3)
    assert res.converged
    assert res.backtracks <= 3
