"""Experiment 2 - build the Google matrix for ca-HepPh, validate it, benchmark it.

``ca-HepPh`` is the arXiv high-energy-physics co-authorship network: 12,008
authors, an edge whenever two of them wrote a paper together.  It is
**undirected**, which is the property that matters here -- section 5 shows its
Google matrix has a real spectrum, which is precisely the condition the base
paper's acceleration theory assumes.

The matrix is validated two independent ways before any timing is reported,
because a wrong operator would invalidate everything downstream:

  1. agreement with ``networkx.pagerank``, on a tolerance that scales with n
     because NetworkX stops at ``n * tol`` and is the less accurate of the two;
  2. the fixed-point residual ``||Gx - x||_1``, which needs no other library to
     be trusted.

    python experiment/exp2_pagerank.py
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from _common import GRAPH, MAX_ITER, TOL, banner, result_path, save_figure
from _methods import run_all_methods
from _style import STYLE, markevery, use_style
from src.datasets import load_snap_edges
from src.google_matrix import build_google_matrix, normalise_ranking
from src.iterations import asymptotic_rate, optimal_beta, power_iteration, predicted_speedup
from src.metrics import l1_error, precision_at_k


def validate(src, dst, node_ids, G):
    """Cross-check the operator against NetworkX and against itself."""
    import networkx as nx

    G.reset()
    ours = normalise_ranking(power_iteration(G, tol=1e-14, max_iter=100_000).x)
    G.reset()
    fixed_point = float(np.abs(G.matvec(ours) - ours).sum())

    mask = src != dst
    nxg = nx.DiGraph()
    nxg.add_nodes_from(node_ids.tolist())
    nxg.add_edges_from(zip(src[mask].tolist(), dst[mask].tolist()))
    ref = nx.pagerank(nxg, alpha=G.d, tol=1e-13, max_iter=1000)
    ref = np.array([ref[int(v)] for v in node_ids], dtype=float)
    err = float(np.abs(ours - ref).sum())
    tol = max(1e-9, 20 * G.n * 1e-13)
    return (err < tol and fixed_point < 1e-12), err, tol, fixed_point


def measure_lambda2(G, k: int = 8, tol: float = 1e-8) -> float:
    from scipy.sparse.linalg import LinearOperator, eigs
    op = LinearOperator((G.n, G.n), matvec=G.matvec, dtype=float)
    vals = np.asarray(eigs(op, k=min(k, G.n - 2), which="LM", tol=tol,
                           v0=np.ones(G.n), return_eigenvectors=False))
    return float(abs(vals[np.argsort(-np.abs(vals))][1]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--damping", type=float, default=0.85)
    args = ap.parse_args()
    use_style()
    import matplotlib.pyplot as plt

    d = args.damping
    banner(f"{GRAPH}: building and validating the Google matrix  (d = {d})")
    src, dst = load_snap_edges(GRAPH)
    G, node_ids = build_google_matrix(src, dst, d=d, name=GRAPH)
    print(f"  {G!r}")
    print(f"  undirected: the link structure is symmetric, "
          f"{int(G.dangling.sum())} dangling node(s) of {G.n:,}")

    ok, err, tol, fp = validate(src, dst, node_ids, G)
    print(f"  [{'PASS' if ok else 'FAIL'}] matches networkx.pagerank "
          f"(l1 error {err:.2e}, tolerance {tol:.1e})")
    print(f"           fixed-point residual ||Gx-x||_1 = {fp:.2e}")

    lam2 = measure_lambda2(G)
    print(f"  measured |lambda_2(G)| = {lam2:.6f}   (d = {d}, ratio {lam2 / d:.4f})")
    print(f"  -> lambda_1 = 1 and |lambda_2| = d, so the optimal static parameter is")
    print(f"     beta = d^2/4 = {optimal_beta(d):.6f}, known in closed form with no eigensolve.")
    print(f"  theory predicts {predicted_speedup(d):.2f}x "
          f"(rate {d:.3f} -> {asymptotic_rate(d):.4f})")

    banner(f"{GRAPH}: the three published methods at d = {d}")
    res = run_all_methods(G, d, tol=TOL, max_iter=MAX_ITER)
    base = res["power"].matvecs
    reference = normalise_ranking(res["power"].x)
    rows = []
    for name, out in res.items():
        sp = base / out.matvecs
        print(f"  {name:>8s}: {out.matvecs:5d} matvecs  {out.time * 1e3:7.1f} ms  "
              f"residual {out.final_residual():.2e}  speed-up {sp:5.2f}x  "
              f"agreement l1={l1_error(out.x, reference):.1e}  "
              f"top100={precision_at_k(out.x, reference, 100):.3f}")
        rows.append(dict(graph=GRAPH, n=G.n, nnz=int(G.P.nnz), d=d, method=name,
                         matvecs=out.matvecs, iterations=out.n_iter, time_s=out.time,
                         converged=out.converged, residual=out.final_residual(),
                         speedup=sp, l1_agreement=l1_error(out.x, reference),
                         precision_at_100=precision_at_k(out.x, reference, 100),
                         lambda2_measured=lam2, predicted_speedup=predicted_speedup(d),
                         dangling=int(G.dangling.sum())))

    pd.DataFrame(rows).to_csv(result_path("exp2_pagerank.csv"), index=False)
    print("\nwrote result/exp2_pagerank.csv")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.7))
    for name, out in res.items():
        y = np.asarray(out.residual)
        ax[0].semilogy(np.arange(1, y.size + 1), y,
                       markevery=markevery(y.size), **STYLE[name])
    ax[0].set_xlabel("matrix-vector products")
    ax[0].set_ylabel(r"$\|Gx_k-\nu_k x_k\|_2$")
    ax[0].set_title(f"{GRAPH} PageRank, $d={d}$"); ax[0].legend()

    b = np.asarray(res["dynamic"].beta)
    ax[1].plot(np.arange(1, b.size + 1), b, color=STYLE["dynamic"]["color"],
               label=r"$\beta_k$ chosen by Alg. 3.1")
    ax[1].axhline(optimal_beta(d), color="black", ls="--", lw=1.2,
                  label=r"$\beta_{opt}=d^2/4$")
    ax[1].axhline(0.25, color="#C00000", ls=":", lw=1.2,
                  label=r"$\lambda_1^2/4$: no convergence above")
    ax[1].set_xlabel("iteration"); ax[1].set_ylabel(r"$\beta_k$")
    ax[1].set_ylim(0, 0.28)
    ax[1].set_title(r"The self-tuned $\beta_k$ stays safely below the barrier")
    ax[1].legend(fontsize=7.5)
    save_figure(fig, "exp2_pagerank")

    banner("GATE")
    print(f"  [{'PASS' if ok else 'FAIL'}] Google matrix validated")
    print(f"  [{'PASS' if res['dynamic'].converged else 'FAIL'}] "
          f"Algorithm 3.1 converged, unmodified")
    print(f"  [{'PASS' if base / res['dynamic'].matvecs > 1 else 'FAIL'}] "
          f"dynamic momentum beats plain power iteration "
          f"({base / res['dynamic'].matvecs:.2f}x)")
    return 0 if ok and res["dynamic"].converged else 1


if __name__ == "__main__":
    raise SystemExit(main())
