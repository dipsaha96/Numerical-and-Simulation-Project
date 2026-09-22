"""The method set compared on every PageRank problem, in one place.

Keeping this here rather than in each experiment script guarantees that the
benchmark, the damping sweep and the ranking-quality study all compare exactly
the same five configurations with exactly the same parameters.

The five are:

``power``
    Algorithm 1.1.  The baseline every speedup is measured against.
``static_d``
    Algorithm 1.2 with ``beta = d^2/4``.  For the Google matrix this is the
    paper's optimal parameter evaluated *exactly and for free*: ``lambda_1 = 1``
    by stochasticity, and Haveliwala and Kamvar (2003) give ``|lambda_2| <= d``
    with equality on graphs whose stochastic-completed link matrix has two or
    more irreducible closed subsets -- which ``exp2`` confirms holds to six
    decimal places on the citation and web graphs tested here.  No eigensolve is
    needed, so this is a stronger baseline than the paper could use.
``dynamic_paper``
    Algorithm 3.1 exactly as printed.
``dynamic_safe``
    Algorithm 3.1 with the three safeguards developed in ``src.iterations``:
    the ``lambda_1 = 1`` cap, the ``r <= d`` bound, and residual-triggered
    backtracking with slow growth.
``dynamic_guard``
    The same, plus the rate guard, which retreats to the power iteration as soon
    as momentum stops paying for itself.
"""

from __future__ import annotations

from .iterations import dynamic_momentum, optimal_beta, power_iteration, static_momentum

__all__ = ["PAGERANK_METHODS", "run_methods"]

# Safeguard settings shared by both safeguarded variants.  `backtrack_window=8`
# is short enough to catch the divergence early and long enough not to fire on
# ordinary residual wobble; `backtrack_grow=1.06` with `grow_patience=4` walks
# the ceiling back up slowly enough to stay below the stability boundary.
_SAFE = dict(backtrack_window=8, backtrack_grow=1.06, grow_patience=4,
             max_backtracks=20)


def _runner(name: str):
    def run(G, d: float, tol: float, max_iter: int, **kw):
        if name == "power":
            return power_iteration(G, tol=tol, max_iter=max_iter, **kw)
        if name == "static_d":
            return static_momentum(G, beta=optimal_beta(d), tol=tol,
                                   max_iter=max_iter, **kw)
        if name == "dynamic_paper":
            return dynamic_momentum(G, tol=tol, max_iter=max_iter, **kw)
        if name == "dynamic_safe":
            return dynamic_momentum(G, tol=tol, max_iter=max_iter, lambda1=1.0,
                                    r_max=d, **_SAFE, **kw)
        if name == "dynamic_guard":
            return dynamic_momentum(G, tol=tol, max_iter=max_iter, lambda1=1.0,
                                    r_max=d, rate_guard=True, **_SAFE, **kw)
        raise KeyError(name)
    return run


PAGERANK_METHODS = {name: _runner(name) for name in
                    ("power", "static_d", "dynamic_paper",
                     "dynamic_safe", "dynamic_guard")}


def run_methods(G, d: float, tol: float = 1e-12, max_iter: int = 80_000,
                methods=None, **kw) -> dict:
    """Run the method set on one Google matrix and return ``{name: result}``."""
    names = methods or list(PAGERANK_METHODS)
    out = {}
    for name in names:
        G.reset()
        out[name] = PAGERANK_METHODS[name](G, d, tol, max_iter, **kw)
    return out
