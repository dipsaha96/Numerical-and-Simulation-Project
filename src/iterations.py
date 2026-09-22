"""Power iteration and its momentum-accelerated variants.

This module is deliberately ignorant of PageRank.  Every routine here takes an
arbitrary linear operator ``A`` (anything exposing ``A @ x`` -- a dense array, a
SciPy sparse matrix, or a :class:`scipy.sparse.linalg.LinearOperator`) and
returns the dominant eigenpair.  That is what lets the SuiteSparse reproduction
experiments and the Google-matrix experiments share a single code path.

Three methods are provided, all costing exactly one matrix-vector product per
iteration:

``power_iteration``
    Algorithm 1.1 of Austin, Pollock & Zhu (2024).
``static_momentum``
    Algorithm 1.2 -- the heavy-ball acceleration of Xu, He & Gu (2018) with a
    fixed parameter ``beta``.  Optimal at ``beta = lambda_2 ** 2 / 4``, which
    requires a priori spectral knowledge.
``dynamic_momentum``
    Algorithm 3.1 -- the contribution of the paper.  ``beta_k`` is re-estimated
    every iteration from the Rayleigh quotient and the ratio of the last two
    residuals, so no spectral knowledge is needed.

Notation follows the paper exactly (``u``, ``v``, ``x``, ``h``, ``nu``, ``d``,
``rho``, ``r``) so that the code can be read next to Algorithms 1.1/1.2/3.1.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "IterationResult",
    "power_iteration",
    "static_momentum",
    "dynamic_momentum",
    "optimal_beta",
    "asymptotic_rate",
    "predicted_speedup",
]


# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------


@dataclass
class IterationResult:
    """Everything an experiment script needs from a single solver run.

    Attributes
    ----------
    x:
        The converged approximation to the dominant eigenvector, 2-normalised.
    nu:
        Final Rayleigh quotient, i.e. the dominant eigenvalue estimate.
    residual:
        History of ``||A x_k - nu_k x_k||_2``, one entry per iteration.
    beta:
        History of the momentum parameter.  Constant for ``static_momentum``
        and identically zero for ``power_iteration``; recorded for all three so
        the plotting code can stay uniform.
    r_est:
        History of the estimated spectral ratio ``r_k ~ |lambda_2 / lambda_1|``
        (``dynamic_momentum`` only, ``NaN`` elsewhere).
    iterates:
        Optional history of the iterates themselves.  Only populated when
        ``track_iterates=True``; used by the ranking-quality experiment, which
        needs to score the partial ranking at every step.
    matvecs:
        Total number of matrix-vector products consumed, *including* the
        preliminary iterations each momentum method needs to prime itself.
    n_iter:
        Number of main-loop iterations performed.
    time:
        Wall-clock seconds.
    converged:
        Whether the residual tolerance was met before ``max_iter``.
    safeguard_hits:
        Number of iterations on which the ``beta_k < lambda_1^2 / 4`` cap had to
        clip the momentum parameter.
    r_clip_hits:
        Number of iterations on which the estimated spectral ratio ``r_k`` had
        to be clipped to the known bound ``r_max``.
    backtracks:
        Number of times the momentum ceiling was shrunk after the residual
        stagnated.  All three counters are zero for the symmetric problems the
        paper analyses; see ``dynamic_momentum``.
    diverged:
        Set when the run was stopped early because the residual had grown far
        past its best value.  Distinguishes genuine divergence from merely
        running out of iterations, which matters here because the two have very
        different causes and the experiments report them separately.
    """

    x: np.ndarray
    nu: float
    residual: list[float] = field(default_factory=list)
    beta: list[float] = field(default_factory=list)
    r_est: list[float] = field(default_factory=list)
    iterates: list[np.ndarray] = field(default_factory=list)
    matvecs: int = 0
    n_iter: int = 0
    time: float = 0.0
    converged: bool = False
    safeguard_hits: int = 0
    r_clip_hits: int = 0
    backtracks: int = 0
    diverged: bool = False
    method: str = ""

    def final_residual(self) -> float:
        return self.residual[-1] if self.residual else float("nan")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _as_matvec(A):
    """Return a callable performing ``x -> A x`` for any array-like operator."""
    if callable(A) and not hasattr(A, "__matmul__"):
        return A
    return lambda x: A @ x


def _initial_vector(A, v0, n: int | None = None) -> np.ndarray:
    """Normalise the user-supplied starting vector, defaulting to all-ones.

    The paper reports results from ``u0 = (1, 1, ..., 1)^T`` so that runs are
    reproducible, and separately from random vectors; both are supported here.
    """
    if v0 is None:
        if n is None:
            n = A.shape[0]
        v0 = np.ones(n, dtype=float)
    return np.asarray(v0, dtype=float).ravel()


def optimal_beta(lambda2: float) -> float:
    """The asymptotically optimal static momentum parameter ``lambda_2^2 / 4``.

    See Corollary 2.3 of the paper.  For the Google matrix this is simply
    ``d ** 2 / 4``, because ``lambda_1 = 1`` and ``|lambda_2| = d``.
    """
    return float(lambda2) ** 2 / 4.0


def asymptotic_rate(r: float) -> float:
    """Convergence rate ``rho(r) = r / (1 + sqrt(1 - r^2))`` from eq. (2.13).

    This is the rate the momentum iteration attains with the optimal parameter,
    versus the plain power iteration's rate of ``r`` itself.
    """
    r = float(r)
    return r / (1.0 + np.sqrt(max(0.0, 1.0 - r * r)))


def predicted_speedup(r: float) -> float:
    """Theoretical iteration-count speedup ``log(rho(r)) / log(r)``.

    Reaching a tolerance ``eps`` costs ``log(eps)/log(rate)`` iterations, so the
    ratio of the two rates' logarithms is the predicted factor.  For the
    PageRank damping factors of interest this ranges from 3.6x at ``d = 0.85``
    to 44.7x at ``d = 0.999``.
    """
    r = float(r)
    if not 0.0 < r < 1.0:
        return float("nan")
    return np.log(asymptotic_rate(r)) / np.log(r)


# --------------------------------------------------------------------------
# Algorithm 1.1 -- power iteration
# --------------------------------------------------------------------------


def power_iteration(
    A,
    v0=None,
    tol: float = 1e-12,
    max_iter: int = 2000,
    track_iterates: bool = False,
    n: int | None = None,
) -> IterationResult:
    """Algorithm 1.1: the classical power iteration.

    Parameters
    ----------
    A:
        Linear operator; only ``A @ x`` is used.
    v0:
        Starting vector, defaulting to all-ones as in the paper's reproducible
        experiments.
    tol:
        Stop once ``||A x_k - nu_k x_k||_2 < tol``.
    max_iter:
        Iteration cap; the paper uses 2000.
    track_iterates:
        Retain every iterate so that ranking quality can be scored per step.
    """
    matvec = _as_matvec(A)
    v = _initial_vector(A, v0, n)

    t0 = time.perf_counter()
    h = np.linalg.norm(v)
    x = v / h
    v_next = matvec(x)          # v_1
    matvecs = 1

    res: list[float] = []
    iterates: list[np.ndarray] = []
    converged = False
    k = 0

    for k in range(1, max_iter + 1):
        h = np.linalg.norm(v_next)
        x = v_next / h
        v_next = matvec(x)      # the single matrix-vector product
        matvecs += 1
        nu = float(v_next @ x)
        d = float(np.linalg.norm(v_next - nu * x))

        res.append(d)
        if track_iterates:
            iterates.append(x.copy())
        if d < tol:
            converged = True
            break

    elapsed = time.perf_counter() - t0
    return IterationResult(
        x=x,
        nu=float(v_next @ x),
        residual=res,
        beta=[0.0] * len(res),
        r_est=[float("nan")] * len(res),
        iterates=iterates,
        matvecs=matvecs,
        n_iter=k,
        time=elapsed,
        converged=converged,
        method="power",
    )


# --------------------------------------------------------------------------
# Algorithm 1.2 -- static momentum
# --------------------------------------------------------------------------


def static_momentum(
    A,
    beta: float,
    v0=None,
    tol: float = 1e-12,
    max_iter: int = 2000,
    divergence_factor: float | None = 1e3,
    stall_window: int | None = 1000,
    track_iterates: bool = False,
    n: int | None = None,
) -> IterationResult:
    """Algorithm 1.2: power iteration with a fixed momentum parameter.

    ``beta`` should be ``lambda_2 ** 2 / 4`` for the optimal asymptotic rate
    (Corollary 2.3).  Note the hard requirement ``beta < lambda_1 ** 2 / 4``:
    above that threshold every eigenvalue of the augmented matrix has the same
    magnitude ``sqrt(beta)`` and the iteration cannot converge at all.

    Two early stops keep a failing run from consuming the whole budget, and make
    the output say *how* it failed rather than only that it ran out of
    iterations.  ``divergence_factor`` stops once the residual exceeds its best
    value by that factor.  ``stall_window`` stops once that many iterations pass
    with no new best residual, which is the failure mode that actually occurs
    here: with ``beta`` past the stability boundary the residual does not blow
    up, it settles onto a plateau.  On ``web-Stanford`` at ``d = 0.85``,
    ``beta = d^2/4`` plateaus at 0.85 and without this check spends 80,000
    matrix-vector products and 467 seconds confirming it.

    Neither can fire on a converging run: convergence means the best residual
    keeps improving, and a monotone sequence never triggers either test.
    """
    matvec = _as_matvec(A)
    v = _initial_vector(A, v0, n)

    t0 = time.perf_counter()

    # One preliminary power iteration (k = 0) to produce x_0, x_1, h_1.
    h_prev = np.linalg.norm(v)
    x_prev = v / h_prev                 # x_0
    v_next = matvec(x_prev)             # v_1
    matvecs = 1
    h = np.linalg.norm(v_next)
    x = v_next / h                      # x_1
    v_next = matvec(x)                  # v_2
    matvecs += 1
    nu = float(v_next @ x)
    d = float(np.linalg.norm(v_next - nu * x))

    res = [d]
    iterates = [x.copy()] if track_iterates else []
    converged = d < tol
    diverged = False
    best_res = d
    stagnant = 0
    k = 1

    while not converged and k < max_iter:
        k += 1
        # u_{k+1} = v_{k+1} - (beta / h_k) x_{k-1}
        u = v_next - (beta / h) * x_prev
        h_prev, x_prev = h, x
        h = np.linalg.norm(u)
        x = u / h
        v_next = matvec(x)              # the single matrix-vector product
        matvecs += 1
        nu = float(v_next @ x)
        d = float(np.linalg.norm(v_next - nu * x))

        res.append(d)
        if track_iterates:
            iterates.append(x.copy())
        if d < tol:
            converged = True
        if d < best_res:
            best_res, stagnant = d, 0
        else:
            stagnant += 1
        if (divergence_factor is not None and len(res) >= 20
                and d > divergence_factor * best_res):
            diverged = True
            break
        if stall_window is not None and stagnant >= stall_window:
            diverged = True
            break

    elapsed = time.perf_counter() - t0
    return IterationResult(
        x=x,
        nu=nu,
        residual=res,
        beta=[beta] * len(res),
        r_est=[float("nan")] * len(res),
        iterates=iterates,
        matvecs=matvecs,
        n_iter=k,
        time=elapsed,
        converged=converged,
        diverged=diverged,
        method=f"static(beta={beta:.6g})",
    )


# --------------------------------------------------------------------------
# Algorithm 3.1 -- dynamic momentum
# --------------------------------------------------------------------------


def dynamic_momentum(
    A,
    v0=None,
    tol: float = 1e-12,
    max_iter: int = 2000,
    lambda1: float | None = None,
    r_max: float | None = None,
    theta: float = 1.0 - 1e-8,
    backtrack_window: int | None = None,
    backtrack_shrink: float = 0.5,
    backtrack_grow: float = 1.0,
    grow_patience: int = 5,
    max_backtracks: int = 20,
    rate_guard: bool = False,
    divergence_factor: float | None = 1e3,
    stall_window: int | None = 1000,
    r_floor: float = 1e-3,
    track_iterates: bool = False,
    n: int | None = None,
) -> IterationResult:
    """Algorithm 3.1: the paper's dynamic momentum method, with safeguards.

    The core recurrence is the paper's verbatim.  ``beta_k = nu_k^2 r_k^2 / 4``
    where ``r_k`` is obtained by inverting the optimal convergence rate (2.13)
    for ``r`` in terms of the *detected* residual convergence rate
    ``rho_k = d_{k+1}/d_k``::

        rho = r / (1 + sqrt(1 - r^2))    <=>    r = 2 rho / (1 + rho^2)

    Lemma 3.4 shows the inversion is stable, increasingly so as ``r -> 1``.

    With every safeguard left at its default the routine *is* Algorithm 3.1 as
    printed.  The optional arguments exist because Algorithm 3.1 does not
    converge on the Google matrix, for two distinct reasons documented below.

    Parameters
    ----------
    lambda1:
        Known value of (or upper bound on) ``|lambda_1|``.  Section 2 shows the
        iteration cannot converge once ``beta >= lambda_1^2/4``: there every
        eigenvalue of the augmented matrix has magnitude ``sqrt(beta)`` and no
        spectral gap remains.  Theorem 3.1 rules that out for *symmetric* ``A``,
        where ``|nu_k| <= lambda_1`` makes ``beta_k <= lambda_1^2/4`` automatic.
        For PageRank ``G`` is column-stochastic, so ``lambda_1 = 1`` exactly and
        callers simply pass ``lambda1=1.0``.

        The cap has to be taken against this *known* ``lambda_1``, never against
        ``nu_k``.  Capping at ``theta * nu_k^2/4`` looks equivalent, but since
        ``nu_k -> lambda_1`` any ``theta < 1`` eventually clips ``beta_k`` below
        the optimum whenever ``r^2 > theta`` -- exactly the ``r -> 1`` regime the
        method exists to serve.  On ``diag(1000:-1:1)`` a 1% clip degrades the
        rate from 0.9562 to 0.9895 and quadruples the iteration count.
    r_max:
        Known upper bound on ``r = |lambda_2/lambda_1|``.

        Algorithm 3.1 clamps the detected rate with ``min(rho_k, 1)``.  On a
        symmetric matrix the residual decreases monotonically and that clamp is
        inert.  The Google matrix is non-normal, its residual oscillates, and a
        single step with ``d_{k+1} > d_k`` sets ``rho_k = 1``, hence
        ``r_{k+1} = 1`` and ``beta_{k+1} = nu^2/4 -> lambda_1^2/4``: the exact
        value at which convergence stops.  The residual then grows, ``rho_k``
        stays pinned at one and the method never recovers.  Measured on
        ``wiki-Vote`` at ``d = 0.85``: the residual bottoms out near ``1e-3``
        around iteration 20, climbs back to ``1.5e-1``, and 2000 iterations end
        further from the answer than 20 did.

        PageRank supplies the bound for free.  Haveliwala and Kamvar (2003)
        prove ``|lambda_2(G)| <= d`` for every graph, so ``r_max = d`` is always
        valid, never excludes the true ``r``, and holds ``beta_k <= d^2/4``.
    backtrack_window:
        Number of consecutive iterations without a new best residual that
        triggers a backtrack, or ``None`` to disable.

        ``r_max = d`` fixes the ``rho_k = 1`` runaway but not the deeper
        problem.  The paper analyses real spectra, where every subdominant
        eigenvalue with ``lambda^2/4 < beta`` maps to ``|mu| = sqrt(beta)``
        (eq. 2.11).  That identity fails for complex ``lambda``, and the Google
        matrix has plenty.  Writing ``mu_+- = (lambda +- sqrt(lambda^2-4 beta))/2``,
        a purely imaginary ``lambda = i y`` gives
        ``|mu| = (y + sqrt(y^2 + 4 beta))/2``, which *grows* with ``|y|`` and is
        unbounded above by ``sqrt(beta)``.

        The consequence is counter-intuitive and invisible to the paper's
        analysis: the dangerous modes are not the ones near the top of the
        spectrum but small-modulus eigenvalues lying near the imaginary axis.
        On ``wiki-Vote`` at ``d = 0.99`` with ``beta = d^2/4``, eigenvalues of
        modulus ``0.23`` and argument ``~pi/2`` reach ``|mu| = 0.624`` against
        ``|mu_lambda_1| = 0.571`` -- so the augmented iteration converges to the
        wrong eigenmode, and no choice of ``r_max <= d`` can prevent it.

        Backtracking removes the need for any spectral knowledge.  The best
        iterate seen so far is retained; if ``backtrack_window`` iterations pass
        without improving on it, the momentum ceiling is multiplied by
        ``backtrack_shrink``, the iteration restarts from that best iterate, and
        it continues.  Since ``beta -> 0`` degenerates to the plain power
        iteration, which always converges for ``d < 1``, the safeguarded method
        terminates whatever the spectrum looks like -- and in practice it keeps
        most of the acceleration, because it only gives up as much momentum as
        the spectrum actually forces it to.
    backtrack_shrink:
        Factor applied to the momentum ceiling on each backtrack.
    backtrack_grow:
        Factor by which the ceiling is *raised* after ``grow_patience``
        consecutive improvements, capped at ``r_max``.  ``1.0`` disables growth,
        leaving a pure retreat strategy.

        Growth matters because retreat alone is one-directional and overshoots
        downward.  The useful momentum for a non-normal matrix sits at the
        largest ``beta`` that is still stable, and that value cannot be computed
        from ``lambda_2`` -- on ``cit-HepPh`` at ``d = 0.95`` the paper's
        ``beta = d^2/4 = 0.2256`` gives a true convergence rate of ``1.097``
        (divergence), while ``beta ~ 0.184`` gives ``0.897`` against the power
        iteration's ``0.95``.  Halving on stagnation lands near ``0.056`` and
        recovers almost none of that.  Alternating shrink-on-stagnation with
        slow growth-on-progress makes the ceiling hover just below the stability
        boundary, which is the best that can be done without computing the
        complex spectrum.
    rate_guard:
        Retreat as soon as the observed convergence rate is worse than the rate
        the priming power steps achieved, rather than waiting for the residual
        to stagnate.  Off by default.

        The trade-off is sharp and worth stating, because the two settings suit
        different goals.  With the guard on, the method is never materially
        worse than the plain power iteration (measured 0.94x-1.00x on
        ``cit-HepPh`` across ``d = 0.85 ... 0.999``) but gives up the
        acceleration too, since the priming ratio is a pre-asymptotic estimate
        that momentum cannot beat early on and the method retreats to
        ``beta = 0`` before it has had a chance.  With the guard off, retreat
        waits for genuine stagnation: ``cit-HepPh`` then reaches 2.3x at
        ``d = 0.95`` and 4.1x at ``d = 0.99``, at the cost of being about 20%
        slower than the power iteration at ``d = 0.85``, where no useful
        momentum exists for this spectrum.
    max_backtracks:
        Budget of backtracks.  Once it is exhausted the ceiling is set to zero
        and the iteration continues as a plain power iteration.  This is what
        turns the safeguard into a guarantee: momentum is only ever used while
        it is observably paying for itself, and the worst case is the cost of
        the power iteration plus ``3 * max_backtracks`` matrix-vector products.
        Without the budget the method can thrash -- at ``d = 0.999`` on
        ``cit-HepPh`` an unbudgeted version spent 7240 backtracks and never
        converged.
    divergence_factor:
        Stop and report ``diverged=True`` once the residual exceeds its best
        value by this factor.
    stall_window:
        Stop and report ``diverged=True`` after this many iterations with no new
        best residual -- the plateau failure mode, which no growth test catches.

        Both are consulted only when backtracking is disabled: with backtracking
        on, a stalled or growing residual is the signal the safeguard exists to
        act on, not a reason to give up.
    r_floor:
        Ceiling below which momentum is abandoned entirely; the iteration is
        then plain power iteration.
    theta:
        Slack on the ``lambda1`` cap, kept a hair below one so the strict
        inequality ``beta_k < lambda_1^2/4`` holds while leaving the optimal
        ``beta = lambda_2^2/4`` untouched for every ``r < 1 - 5e-9``.
    """
    matvec = _as_matvec(A)
    v_start = _initial_vector(A, v0, n)

    t0 = time.perf_counter()
    state = {"matvecs": 0}

    def prime(start: np.ndarray):
        """Two preliminary power iterations, as Algorithm 3.1 requires.

        Returns the state the momentum loop needs: ``(h_prev, x_prev, h, x,
        v_next, nu, d)`` together with the residuals of both priming steps.
        """
        h0 = np.linalg.norm(start)
        x0 = start / h0
        v1 = matvec(x0)
        h1 = np.linalg.norm(v1)
        x1 = v1 / h1
        v2 = matvec(x1)
        nu1 = float(np.real(v2 @ x1))
        d1 = float(np.linalg.norm(v2 - nu1 * x1))
        h2 = np.linalg.norm(v2)
        x2 = v2 / h2
        v3 = matvec(x2)
        nu2 = float(np.real(v3 @ x2))
        d2 = float(np.linalg.norm(v3 - nu2 * x2))
        state["matvecs"] += 3
        return (h1, x1, h2, x2, v3, nu2, d1, d2)

    h_prev, x_prev, h, x, v_next, nu, d1, d = prime(v_start)

    # The ratio of the two priming residuals estimates the rate the plain power
    # iteration achieves here.  Momentum has to beat it to be worth using, and
    # this reference costs nothing extra -- the priming steps are required
    # anyway.
    power_rate_ref = min(d / d1, 1.0) if d1 > 0 else 1.0

    res: list[float] = [d1, d]
    betas: list[float] = [0.0, 0.0]
    r_hist: list[float] = [float("nan"), min(d / d1, 1.0) if d1 > 0 else 1.0]
    iterates: list[np.ndarray] = [x_prev.copy(), x.copy()] if track_iterates else []

    r = r_hist[-1]
    d_prev = d

    best_res = min(d1, d)
    best_x = (x_prev if d1 <= d else x).copy()
    stagnant = 0
    improving = 0

    cap = theta * float(lambda1) ** 2 / 4.0 if lambda1 is not None else np.inf
    r_bound = float(r_max) if r_max is not None else np.inf

    safeguard_hits = 0
    r_clip_hits = 0
    backtracks = 0
    diverged = False
    converged = best_res < tol
    k = 2

    while not converged and state["matvecs"] < max_iter:
        k += 1

        r_used = min(r, r_bound)
        if r > r_bound:
            r_clip_hits += 1
        beta = nu * nu * r_used * r_used / 4.0
        if beta > cap:
            beta = cap
            safeguard_hits += 1

        # u_{k+1} = v_{k+1} - (beta_k / h_k) x_{k-1}
        u = v_next - (beta / h) * x_prev
        h_prev, x_prev = h, x
        h = np.linalg.norm(u)
        x = u / h
        v_next = matvec(x)          # the single matrix-vector product
        state["matvecs"] += 1
        nu = float(np.real(v_next @ x))
        d = float(np.linalg.norm(v_next - nu * x))

        res.append(d)
        betas.append(beta)
        if track_iterates:
            iterates.append(x.copy())

        # rho_k = min(d_{k+1}/d_k, 1);  r_{k+1} = 2 rho_k / (1 + rho_k^2)
        rho = min(d / d_prev, 1.0) if d_prev > 0 else 1.0
        r = 2.0 * rho / (1.0 + rho * rho)
        r_hist.append(r)
        d_prev = d

        if d < tol:
            converged = True
            break

        # ---- backtracking safeguard ----
        # Trigger 2: momentum is converging, but more slowly than the plain
        # power iteration would.  Stagnation alone cannot detect this -- the
        # residual still falls monotonically -- yet it is the common case at
        # moderate damping, where beta = d^2/4 badly overshoots the largest
        # stable value.
        underperforming = False
        if (rate_guard and backtrack_window is not None and r_bound > 0.0
                and len(res) >= 2 * backtrack_window):
            window = np.asarray(res[-backtrack_window:], dtype=float)
            start = float(res[-backtrack_window - 1])
            if start > 0 and window[-1] > 0:
                observed = (window[-1] / start) ** (1.0 / backtrack_window)
                underperforming = observed > power_rate_ref

        # `stagnant` must advance whether or not backtracking is enabled: with
        # it disabled the counter is what the stall test below reads, and
        # leaving it pinned at zero let a stalled Algorithm 3.1 run out the full
        # budget (80,000 matrix-vector products on web-Stanford).
        if d < best_res and not underperforming:
            best_res, best_x, stagnant = d, x.copy(), 0
            improving += 1
            if (backtrack_grow > 1.0 and r_max is not None
                    and improving >= grow_patience and 0.0 < r_bound < r_max
                    and backtracks < max_backtracks // 2):
                r_bound = min(float(r_max), r_bound * backtrack_grow)
                improving = 0
        else:
            stagnant += 1

        if backtrack_window is None:
            if (divergence_factor is not None and len(res) >= 20
                    and d > divergence_factor * best_res):
                diverged = True
                break
            if stall_window is not None and stagnant >= stall_window:
                diverged = True
                break

        if backtrack_window is not None and r_bound > 0.0 and d >= best_res:
            # Once the ceiling reaches zero the iteration is a plain power
            # iteration, whose residual may still plateau briefly.  Backtracking
            # must be off by then: there is no momentum left to withdraw, and
            # re-priming from the same best iterate would loop forever, burning
            # three matrix-vector products at a time without making progress.
            if underperforming:
                stagnant = max(stagnant, backtrack_window)
            if stagnant >= backtrack_window:
                backtracks += 1
                stagnant = 0
                improving = 0
                r_bound = min(r_bound, r_used) * backtrack_shrink
                if r_bound < r_floor or backtracks >= max_backtracks:
                    r_bound = 0.0      # give up on momentum: plain power iteration
                h_prev, x_prev, h, x, v_next, nu, _, d = prime(best_x)
                r = 0.0 if r_bound == 0.0 else min(r, r_bound)
                d_prev = d
                if d < best_res:
                    best_res, best_x = d, x.copy()
                if d < tol:
                    converged = True
                    break

    elapsed = time.perf_counter() - t0
    return IterationResult(
        x=x,
        nu=nu,
        residual=res,
        beta=betas,
        r_est=r_hist,
        iterates=iterates,
        matvecs=state["matvecs"],
        n_iter=k,
        time=elapsed,
        converged=converged,
        safeguard_hits=safeguard_hits,
        r_clip_hits=r_clip_hits,
        backtracks=backtracks,
        diverged=diverged,
        method="dynamic",
    )
