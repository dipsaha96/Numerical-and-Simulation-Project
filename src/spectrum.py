"""Spectral diagnostics for the momentum-accelerated iteration.

The momentum iteration (1.2) is a power iteration on the augmented matrix

    A_beta = [[A, -beta I], [I, 0]]

whose eigenvalues are, for each eigenvalue ``lambda`` of ``A``,

    mu_+- = ( lambda +- sqrt(lambda^2 - 4 beta) ) / 2                     (2.7)

so the iteration converges at the rate ``max_{l >= 2} |mu(lambda_l)| /
|mu(lambda_1)|``.

For a **real** spectrum this collapses to the tidy formula the paper works with:
every ``lambda`` with ``lambda^2/4 < beta`` maps to a complex conjugate pair of
modulus exactly ``sqrt(beta)`` (eq. 2.11), so subdominant modes can never be
worse than ``sqrt(beta)`` and ``beta = lambda_2^2/4`` is optimal.

For a **complex** spectrum that identity fails, and it fails upward.  Writing
``lambda = i y`` for a purely imaginary eigenvalue,

    |mu| = ( |y| + sqrt(y^2 + 4 beta) ) / 2,

which exceeds ``sqrt(beta)`` for every ``y != 0`` and grows without bound in
``|y|``.  The consequence is the opposite of the intuition the real case builds:
the modes that destabilise the iteration are not the ones near the top of the
spectrum but *small-modulus* eigenvalues lying near the imaginary axis.

The Google matrix is non-normal and has such eigenvalues, which is why the
paper's parameter choice does not transfer to PageRank.  This module computes
the quantities needed to show that quantitatively.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "augmented_eigenvalues",
    "mu_magnitude",
    "convergence_rate",
    "rate_curve",
    "best_beta",
    "stability_limit",
    "sample_spectrum",
    "imaginary_axis_limit",
]


def augmented_eigenvalues(lam, beta: float):
    """The pair ``mu_+-`` of eq. (2.7) for eigenvalue(s) ``lam``."""
    lam = np.asarray(lam, dtype=complex)
    disc = np.sqrt(lam * lam - 4.0 * beta + 0j)
    return (lam + disc) / 2.0, (lam - disc) / 2.0


def mu_magnitude(lam, beta: float) -> np.ndarray:
    """``max(|mu_+|, |mu_-|)``, the magnitude that governs convergence."""
    mp, mm = augmented_eigenvalues(lam, beta)
    return np.maximum(np.abs(mp), np.abs(mm))


def convergence_rate(spectrum, beta: float, lambda1: complex = 1.0) -> float:
    """Asymptotic rate of the momentum iteration given a sampled spectrum.

    ``spectrum`` must contain the subdominant eigenvalues only.  A returned
    value ``>= 1`` means the iteration does not converge: some subdominant mode
    of the augmented matrix is at least as large as the dominant one, so the
    iteration aligns with the wrong eigenvector.
    """
    sub = np.asarray(spectrum, dtype=complex)
    if sub.size == 0:
        return 0.0
    m1 = float(mu_magnitude(np.asarray([lambda1], dtype=complex), beta)[0])
    return float(mu_magnitude(sub, beta).max() / m1)


def rate_curve(spectrum, betas, lambda1: complex = 1.0) -> np.ndarray:
    """``convergence_rate`` evaluated over an array of ``beta``."""
    return np.array([convergence_rate(spectrum, float(b), lambda1) for b in betas])


def best_beta(spectrum, lambda1: complex = 1.0, n_grid: int = 2000,
              beta_max: float | None = None) -> tuple[float, float]:
    """Grid-search the ``beta`` minimising the true rate, and that rate.

    This is the correct generalisation of ``beta = lambda_2^2/4`` to a
    non-normal matrix.  The two agree when the spectrum is real; they can differ
    enormously otherwise.
    """
    if beta_max is None:
        beta_max = float(abs(lambda1)) ** 2 / 4.0
    betas = np.linspace(0.0, beta_max * (1 - 1e-9), n_grid)
    rates = rate_curve(spectrum, betas, lambda1)
    i = int(np.argmin(rates))
    return float(betas[i]), float(rates[i])


def stability_limit(spectrum, lambda1: complex = 1.0, n_grid: int = 4000,
                    beta_max: float | None = None) -> float:
    """Largest ``beta`` for which the iteration still converges.

    Returns ``0.0`` when no positive ``beta`` is stable.
    """
    if beta_max is None:
        beta_max = float(abs(lambda1)) ** 2 / 4.0
    betas = np.linspace(0.0, beta_max * (1 - 1e-9), n_grid)
    rates = rate_curve(spectrum, betas, lambda1)
    stable = betas[rates < 1.0]
    return float(stable.max()) if stable.size else 0.0


def imaginary_axis_limit(d: float) -> float:
    """Worst-case-safe ``beta`` against a purely imaginary eigenvalue of modulus ``d``.

    Solves ``d + sqrt(d^2 + 4 beta) = 1 + sqrt(1 - 4 beta)``, the point at which
    a hypothetical eigenvalue ``lambda = i d`` of the Google matrix would tie
    with the dominant mode.  This is a bound over the whole disk
    ``|lambda| <= d`` and is therefore pessimistic for any particular graph, but
    it shows how little momentum is safe in the absence of spectral information:
    at ``d = 0.99`` it permits ``beta = 0.00497`` against the paper's
    ``d^2/4 = 0.245``.
    """
    from scipy.optimize import brentq

    def f(b: float) -> float:
        return d + np.sqrt(d * d + 4 * b) - 1.0 - np.sqrt(max(0.0, 1.0 - 4 * b))

    if f(1e-12) >= 0:
        return 0.0
    return float(brentq(f, 1e-12, 0.25 - 1e-9))


def sample_spectrum(op, n: int, k: int = 60, tol: float = 1e-6,
                    which=("LM", "LI", "SI")) -> np.ndarray:
    """Sample the eigenvalues that matter for the rate, dominant one removed.

    A plain ``which="LM"`` call is not enough here.  Because ``|mu(lambda)|``
    depends on the *argument* of ``lambda`` and not only its modulus, the
    eigenvalues that destabilise the iteration can sit well down the spectrum,
    and a largest-magnitude search never returns them.  Sampling the extremes of
    the imaginary part as well (``"LI"``, ``"SI"``) catches them.

    The result is a sample, not the full spectrum, so the rates computed from it
    are lower bounds on the true rate.  That is the conservative direction for
    the conclusion drawn here -- a sampled rate above one proves divergence,
    whereas a sampled rate below one does not prove convergence.
    """
    from scipy.sparse.linalg import LinearOperator, eigs

    if not isinstance(op, LinearOperator):
        matvec = op.matvec if hasattr(op, "matvec") else (lambda x: op @ x)
        op = LinearOperator((n, n), matvec=matvec, dtype=complex)

    k = min(k, n - 2)
    found: list[np.ndarray] = []
    for w in which:
        try:
            found.append(np.asarray(eigs(op, k=k, which=w, tol=tol,
                                         v0=np.ones(n),
                                         return_eigenvectors=False)))
        except Exception:                       # noqa: BLE001 -- ARPACK may fail
            continue
    if not found:
        return np.zeros(0, dtype=complex)
    vals = np.concatenate(found)
    # Drop the dominant eigenvalue (lambda_1 = 1 for a stochastic matrix).
    return vals[np.abs(vals - 1.0) > 1e-8]
