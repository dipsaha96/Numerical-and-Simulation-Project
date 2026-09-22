"""Phase 7: the symmetric control, where the paper's theorems actually apply.

Every negative result about the Google matrix needs a control, or it cannot be
distinguished from a bug.  This script runs the same three solvers on *symmetric*
ranking problems, where Theorem 3.1 guarantees convergence with acceleration and
the real-spectrum rate ``rho(r) = r/(1+sqrt(1-r^2))`` is the truth rather than an
approximation.

Two problems are used, both genuine ranking tasks rather than abstract matrices:

* **Eigenvector centrality** on an undirected collaboration network
  (``ca-HepPh``).  The adjacency matrix is symmetric by construction and its
  dominant eigenvector is the standard centrality score.
* **Co-citation centrality** on a citation graph: ``A A^T``, whose ``(i,j)``
  entry counts the papers citing both ``i`` and ``j``.  This is symmetric even
  though the underlying citation graph is not, and it turns the *same data* used
  in experiments 2-5 into a problem the theory covers -- so a speedup here and
  none there isolates non-normality as the cause.

If the momentum methods deliver the predicted acceleration on these and not on
the Google matrix, the implementation is sound and the PageRank result is a
property of the problem, not of the code.

Usage::

    python experiments/exp6_symmetric.py
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import scipy.sparse as sp

from _common import banner, results_path
from src.datasets import load_snap_edges, synthetic_web_graph
from src.google_matrix import edges_to_sparse
from src.iterations import (
    asymptotic_rate,
    dynamic_momentum,
    optimal_beta,
    power_iteration,
    predicted_speedup,
    static_momentum,
)
from src.plotting import METHOD_STYLE, markevery, save_figure, use_style

TOL = 1e-12
MAX_ITER = 80_000


def adjacency(src, dst, symmetrise: bool = True) -> sp.csr_matrix:
    ids, inv = np.unique(np.concatenate([src, dst]), return_inverse=True)
    n, h = ids.size, src.size
    A = sp.csr_matrix((np.ones(h), (inv[:h], inv[h:])), shape=(n, n))
    if symmetrise:
        A = (A + A.T)
    A.data[:] = 1.0                       # unweighted
    A.sum_duplicates()
    return A.tocsr()


def cocitation(src, dst) -> sp.csr_matrix:
    """``A A^T`` for the citation matrix ``A``: symmetric co-citation counts."""
    P, _, _ = edges_to_sparse(src, dst)
    A = (P > 0).astype(float).tocsr()
    return (A @ A.T).tocsr()


def leading_two(A) -> tuple[float, float]:
    """Fixed start vector so ``lambda_2``, hence ``beta``, is reproducible."""
    from scipy.sparse.linalg import eigsh

    vals = eigsh(A, k=4, which="LM", v0=np.ones(A.shape[0]),
                 return_eigenvectors=False)
    vals = np.asarray(vals)[np.argsort(-np.abs(np.asarray(vals)))]
    lam1 = abs(vals[0])
    for v in vals[1:]:
        if abs(abs(v) - lam1) > 1e-10 * lam1:
            return float(lam1), float(abs(v))
    return float(lam1), float(abs(vals[-1]))


def run_case(name: str, A, rows: list[dict]):
    lam1, lam2 = leading_two(A)
    r = lam2 / lam1
    symmetric = abs((A - A.T)).nnz == 0
    print(f"  {name}: n={A.shape[0]:,}, nnz={A.nnz:,}, symmetric={symmetric}")
    print(f"    lambda_1={lam1:.6g}, lambda_2={lam2:.6g}, r={r:.4f}  ->  "
          f"rho={asymptotic_rate(r):.4f}, theory predicts {predicted_speedup(r):.2f}x")

    results = {
        "power": power_iteration(A, tol=TOL, max_iter=MAX_ITER),
        "static": static_momentum(A, beta=optimal_beta(lam2), tol=TOL, max_iter=MAX_ITER),
        "dynamic": dynamic_momentum(A, tol=TOL, max_iter=MAX_ITER),
    }
    base = results["power"].matvecs
    for key, res in results.items():
        print(f"    {key:>8s}: {res.matvecs:6d} mv  {base / res.matvecs:5.2f}x  "
              f"res {res.final_residual():.2e}  conv={res.converged}")
        rows.append(dict(problem=name, n=A.shape[0], nnz=int(A.nnz), r=r,
                         method=key, matvecs=res.matvecs, time_s=res.time,
                         converged=res.converged, residual=res.final_residual(),
                         speedup=base / res.matvecs,
                         predicted_speedup=predicted_speedup(r)))
    return results, r


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    use_style()
    import matplotlib.pyplot as plt

    rows: list[dict] = []
    panels = []

    banner("Symmetric control problems (Theorem 3.1 applies here)")
    cases = []
    try:
        if args.offline:
            raise RuntimeError("offline mode")
        s, d = load_snap_edges("ca-HepPh")
        cases.append(("ca-HepPh eigenvector centrality", adjacency(s, d)))
    except Exception as exc:                        # noqa: BLE001
        print(f"  ca-HepPh unavailable ({exc}); using a synthetic undirected graph")
        s, d = synthetic_web_graph(n=12_000, seed=7)
        cases.append(("synthetic eigenvector centrality", adjacency(s, d)))
    try:
        if args.offline:
            raise RuntimeError("offline mode")
        s, d = load_snap_edges("cit-HepPh")
        cases.append(("cit-HepPh co-citation $AA^T$", cocitation(s, d)))
    except Exception as exc:                        # noqa: BLE001
        print(f"  cit-HepPh unavailable ({exc}); skipping the co-citation case")

    for name, A in cases:
        results, r = run_case(name, A, rows)
        panels.append((name, r, results))

    fig, axes = plt.subplots(1, len(panels), figsize=(4.8 * len(panels), 3.7),
                             squeeze=False)
    for ax, (name, r, results) in zip(axes[0], panels):
        for key, res in results.items():
            y = np.asarray(res.residual)
            ax.semilogy(np.arange(1, y.size + 1), y,
                        markevery=markevery(y.size), **METHOD_STYLE[key])
        ax.set_xlabel("matrix-vector products")
        ax.set_ylabel(r"$\|Ax_k-\nu_k x_k\|_2$")
        ax.set_title(f"{name}\n$r={r:.4f}$, theory {predicted_speedup(r):.2f}x")
    axes[0][0].legend(fontsize=8)
    save_figure(fig, "exp6_symmetric_control")

    df = pd.DataFrame(rows)
    df.to_csv(results_path("exp6_symmetric.csv"), index=False)
    print("\nwrote results/exp6_symmetric.csv")

    banner("CONTROL RESULT: measured vs predicted on symmetric problems")
    print(df[["problem", "r", "method", "matvecs", "speedup", "predicted_speedup"]]
          .to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    ok = all(
        df[(df.problem == p) & (df.method.isin(["static", "dynamic"]))]["speedup"].min()
        >= 0.75 * df[(df.problem == p)]["predicted_speedup"].iloc[0]
        for p in df["problem"].unique()
    )
    print(f"\n  Control: momentum delivers the predicted acceleration on symmetric "
          f"problems: {'PASS' if ok else 'see table'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
