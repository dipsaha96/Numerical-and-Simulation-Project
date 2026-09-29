"""The three algorithms of the base paper, with nothing added.

``src.iterations`` also implements the safeguards developed elsewhere in this
repository.  This module exists to make it impossible for them to leak into
this suite: every call below passes the arguments that switch them off, so what
runs is Algorithm 1.1, 1.2 and 3.1 exactly as printed in

    Austin, Pollock & Zhu (2024), Numer. Linear Algebra Appl. 31(6), e2584.

Specifically, ``dynamic_momentum`` is called with

    lambda1=None            no cap on beta from a known lambda_1
    r_max=None              no bound on the estimated spectral ratio
    backtrack_window=None   no retreat when the residual stagnates
    rate_guard=False        no retreat when momentum underperforms
    divergence_factor=None  no early stop on a growing residual
    stall_window=None       no early stop on a flat residual

The last two only decide when to give up, not what the iteration computes, but
they are disabled anyway so that a reported iteration count is the honest cost
of running the published algorithm to the tolerance.
"""

from __future__ import annotations

from src.iterations import dynamic_momentum, optimal_beta, power_iteration, static_momentum

__all__ = ["PURE_METHODS", "run_all_methods", "METHOD_LABEL"]

METHOD_LABEL = {
    "power": "Power iteration (Alg. 1.1)",
    "static": r"Static momentum, $\beta=\lambda_2^2/4$ (Alg. 1.2)",
    "dynamic": r"Dynamic momentum, $\beta_k$ (Alg. 3.1)",
}

_PURE = dict(lambda1=None, r_max=None, backtrack_window=None,
             rate_guard=False, divergence_factor=None, stall_window=None)


def run_all_methods(A, lambda2: float, tol: float, max_iter: int, **kw) -> dict:
    """Run all three published methods on one operator.

    ``lambda2`` supplies the static method's optimal parameter.  For a Google
    matrix this is simply the damping factor, because lambda_1 = 1 and
    |lambda_2| = d, so no eigensolver is needed.
    """
    out = {}
    reset = getattr(A, "reset", lambda: None)

    reset()
    out["power"] = power_iteration(A, tol=tol, max_iter=max_iter, **kw)
    reset()
    out["static"] = static_momentum(A, beta=optimal_beta(lambda2), tol=tol,
                                    max_iter=max_iter, divergence_factor=None,
                                    stall_window=None, **kw)
    reset()
    out["dynamic"] = dynamic_momentum(A, tol=tol, max_iter=max_iter, **_PURE, **kw)
    return out


PURE_METHODS = list(METHOD_LABEL)
