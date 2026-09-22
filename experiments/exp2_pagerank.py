"""Phases 2 and 3: build the Google matrix, validate it, and benchmark.

Phase 2 gate
    * The operator must reproduce ``networkx.pagerank``.
    * ``eigs`` must report ``|lambda_2(G)|`` so the a priori parameter
      ``beta = d^2/4`` can be checked against the truth.  Haveliwala and Kamvar
      (2003) prove ``|lambda_2(G)| <= d`` for every graph, with equality when the
      stochastic-completed link matrix has at least two irreducible closed
      subsets.  Whether that holds is a property of the graph, so it is measured
      rather than assumed -- and the answer differs sharply between the citation
      and web graphs (equality to six decimals) and the social graph
      ``wiki-Vote`` (``lambda_2/d = 0.59``, nowhere near tight).

Phase 3
    Run the five methods of ``src.pagerank_methods`` on every graph at
    ``d = 0.85`` and record iterations, matrix-vector products, wall-clock time,
    residual histories and agreement of the resulting rankings.

Usage::

    python experiments/exp2_pagerank.py
    python experiments/exp2_pagerank.py --graphs wiki-Vote cit-HepPh
    python experiments/exp2_pagerank.py --offline
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from _common import banner, results_path
from src.datasets import SNAP_DATASETS, load_snap_edges, synthetic_web_graph
from src.google_matrix import build_google_matrix, normalise_ranking
from src.iterations import asymptotic_rate, optimal_beta, power_iteration, predicted_speedup
from src.metrics import l1_error, precision_at_k
from src.pagerank_methods import run_methods
from src.plotting import METHOD_STYLE, markevery, save_figure, use_style

DEFAULT_GRAPHS = ["wiki-Vote", "cit-HepPh", "web-Stanford"]
TOL = 1e-12
MAX_ITER = 80_000
# NetworkX's PageRank is pure Python: on web-Stanford (282k nodes, 2.3M edges)
# at tol=1e-13 it runs for tens of minutes, which is far too slow to sit in a
# gate.  It is a convenience cross-check on the small graphs only.  The rigorous
# test -- the fixed-point residual ||Gx - x||_1 -- is applied at every size and
# costs a single matrix-vector product.
NETWORKX_LIMIT = 100_000
EIGS_LIMIT = 400_000


# --------------------------------------------------------------------------
# Phase 2 gate
# --------------------------------------------------------------------------


def validate_against_networkx(src, dst, node_ids, G) -> tuple[bool, float | None, float]:
    """Cross-check the Google matrix against NetworkX, and against itself.

    Two independent checks are applied, because neither alone is conclusive.

    The first is agreement with ``networkx.pagerank``.  Its threshold has to
    scale with ``n``: NetworkX stops when the 1-norm change per iteration falls
    below ``n * tol``, so it is the *less* accurate of the two solvers and
    agreement can be no better than its own stopping accuracy.  At ``n = 34546``
    and ``tol = 1e-13`` that floor is ``3.5e-9``, and the measured disagreement
    of ``1.8e-8`` is consistent with NetworkX carrying the error.

    The second is self-contained and independent of any other library: the
    fixed-point residual ``||G x - x||_1`` of our own solution, which must
    vanish for the true PageRank vector whatever NetworkX reports.
    """
    import networkx as nx

    G.reset()
    ours_only = normalise_ranking(power_iteration(G, tol=1e-14, max_iter=10_000).x)
    G.reset()
    fixed_point = float(np.abs(G.matvec(ours_only) - ours_only).sum())
    if G.n > NETWORKX_LIMIT:
        return fixed_point < 1e-12, None, fixed_point

    tol = max(1e-9, 20 * G.n * 1e-13)

    mask = src != dst                       # our builder drops self-loops
    nxg = nx.DiGraph()
    nxg.add_nodes_from(node_ids.tolist())
    nxg.add_edges_from(zip(src[mask].tolist(), dst[mask].tolist()))

    reference = nx.pagerank(nxg, alpha=G.d, tol=1e-13, max_iter=1000)
    ref = np.array([reference[int(v)] for v in node_ids], dtype=float)

    err = float(np.abs(ours_only - ref).sum())
    return (err < tol and fixed_point < 1e-12), err, fixed_point


def measure_second_eigenvalue(G, tol: float = 1e-6, k: int = 8,
                              maxiter: int = 4000) -> float | None:
    """Measure ``|lambda_2(G)|`` -- is the Haveliwala-Kamvar bound tight here?

    On a web graph ``lambda_2 = d`` typically has high multiplicity, one copy per
    irreducible closed subset, and ARPACK converges slowly when the wanted
    eigenvalues are clustered.  Asking for a larger subspace (``k = 8``) gives it
    room to separate them and is much faster than ``k = 3``, and ``tol = 1e-6``
    is ample: the question here is only whether ``|lambda_2|`` equals ``d`` or
    sits well below it, a difference of order 0.1.  ``maxiter`` bounds the cost
    so a hard case degrades to "not measured" rather than hanging the gate.
    """
    from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigs

    if G.n > EIGS_LIMIT:
        return None
    op = LinearOperator((G.n, G.n), matvec=G.matvec, dtype=float)
    try:
        vals = np.asarray(eigs(op, k=min(k, G.n - 2), which="LM", tol=tol,
                               v0=np.ones(G.n), maxiter=maxiter,
                               return_eigenvectors=False))
    except ArpackNoConvergence as exc:
        vals = np.asarray(exc.eigenvalues)
        if vals.size < 2:
            return None
    vals = vals[np.argsort(-np.abs(vals))]
    return float(abs(vals[1]))


def get_graph(key: str, offline: bool):
    if offline or key.startswith("synthetic"):
        src, dst = synthetic_web_graph(n=20_000, seed=abs(hash(key)) % 2**31)
        return src, dst, f"{key} (synthetic)"
    return (*load_snap_edges(key), key)


# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", nargs="*", default=DEFAULT_GRAPHS,
                        help=f"any of {sorted(SNAP_DATASETS)} or 'synthetic'")
    parser.add_argument("--damping", type=float, default=0.85)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    use_style()
    import matplotlib.pyplot as plt

    rows: list[dict] = []
    gate_ok = True
    panels = []
    d = args.damping

    for key in args.graphs:
        banner(f"Graph: {key}")
        try:
            src, dst, label = get_graph(key, args.offline)
        except Exception as exc:                          # noqa: BLE001
            print(f"  unavailable ({type(exc).__name__}: {exc}); using a synthetic graph")
            src, dst, label = get_graph("synthetic", offline=True)

        G, node_ids = build_google_matrix(src, dst, d=d, name=key)
        print(f"  {G!r}")
        print(f"    dangling nodes: {int(G.dangling.sum()):,} "
              f"({100 * G.dangling.mean():.1f}%)")

        lam2 = None
        if not args.skip_validation:
            ok, err, fp = validate_against_networkx(src, dst, node_ids, G)
            nx_note = (f"matches networkx.pagerank (l1 error {err:.3e}, "
                       f"tolerance {max(1e-9, 20 * G.n * 1e-13):.1e}); "
                       if err is not None else
                       f"n > {NETWORKX_LIMIT:,}: NetworkX cross-check skipped; ")
            print(f"  [{'PASS' if ok else 'FAIL'}] {nx_note}"
                  f"fixed-point residual ||Gx-x||_1 = {fp:.2e}")
            gate_ok &= ok
            lam2 = measure_second_eigenvalue(G)
            if lam2 is not None:
                tight = abs(lam2 - d) < 1e-4
                print(f"  [ OK ] |lambda_2(G)| = {lam2:.6f}, d = {d}  -> "
                      f"lambda_2/d = {lam2 / d:.4f}  "
                      f"({'bound TIGHT: beta=d^2/4 is exactly optimal' if tight else 'bound SLACK: d^2/4 overshoots'})")

        r_eff = lam2 if lam2 is not None else d
        print(f"    real-spectrum theory would predict {predicted_speedup(r_eff):.2f}x "
              f"(rate {r_eff:.4f} -> {asymptotic_rate(r_eff):.4f}), "
              f"beta_opt = {optimal_beta(r_eff):.6f}")

        results = run_methods(G, d, tol=TOL, max_iter=MAX_ITER)
        reference = normalise_ranking(results["power"].x)
        base = results["power"].matvecs

        for name, res in results.items():
            # A run that stopped early has no meaningful cost, so it has no
            # meaningful speedup either: reporting `power.matvecs / r.matvecs` for a
            # run that gave up after 1004 of a possible 80,000 matrix-vector
            # products would read as a 19x win for a method that never converged.
            speedup = (base / res.matvecs) if res.converged else float("nan")
            status = ("converged" if res.converged
                      else "DIVERGED" if res.diverged else "no converge")
            print(f"    {name:>14s}: {res.matvecs:6d} mv  {res.time * 1e3:8.1f} ms  "
                  f"res {res.final_residual():.2e}  "
                  f"{speedup:5.2f}x  {status:<11s} " if res.converged else
                  f"{'--':>5s}   {status:<11s} "
                  f"l1={l1_error(res.x, reference):.2e} "
                  f"top100={precision_at_k(res.x, reference, k=min(100, G.n)):.3f}"
                  + (f"  [bt={res.backtracks}]" if res.backtracks else ""))
            rows.append(dict(
                graph=key, n=G.n, nnz=int(G.P.nnz), d=d, method=name,
                matvecs=res.matvecs, iterations=res.n_iter, time_s=res.time,
                converged=res.converged, diverged=res.diverged,
                residual=res.final_residual(),
                speedup=speedup, l1_agreement=l1_error(res.x, reference),
                precision_at_100=precision_at_k(res.x, reference, k=min(100, G.n)),
                safeguard_hits=res.safeguard_hits, r_clip_hits=res.r_clip_hits,
                backtracks=res.backtracks, nu=res.nu,
                dangling=int(G.dangling.sum()),
                lambda2_measured=lam2, lambda2_over_d=(lam2 / d) if lam2 else None,
                predicted_speedup=predicted_speedup(r_eff),
            ))
        panels.append((label, results, lam2))

    # ---- figure ----
    fig, axes = plt.subplots(len(panels), 2, figsize=(10, 3.4 * len(panels)),
                             squeeze=False)
    for row_axes, (label, results, lam2) in zip(axes, panels):
        ax = row_axes[0]
        for name in ("power", "static_d", "dynamic_paper", "dynamic_safe"):
            y = np.asarray(results[name].residual)
            ax.semilogy(np.arange(1, y.size + 1), y,
                        markevery=markevery(y.size), **METHOD_STYLE[name])
        ax.set_xlabel("iteration")
        ax.set_ylabel(r"$\|Gx_k-\nu_k x_k\|_2$")
        ax.set_title(f"{label}  ($d={d}$"
                     + (f", $|\\lambda_2|={lam2:.4f}$)" if lam2 else ")"))
        ax.set_xlim(0, min(1500, max(len(results[m].residual) for m in results)))

        ax = row_axes[1]
        for name in ("dynamic_paper", "dynamic_safe"):
            b = np.asarray(results[name].beta)
            ax.plot(np.arange(1, b.size + 1), b, **METHOD_STYLE[name])
        ax.axhline(optimal_beta(d), color="black", ls="--", lw=1.2,
                   label=r"$\beta=d^2/4$")
        ax.axhline(0.25, color="red", ls=":", lw=1.2,
                   label=r"$\lambda_1^2/4$: no convergence above")
        ax.set_xlabel("iteration")
        ax.set_ylabel(r"$\beta_k$")
        ax.set_xlim(0, min(1500, max(len(results[m].beta) for m in ("dynamic_paper", "dynamic_safe"))))
        ax.set_title("Momentum parameter")
        ax.legend(fontsize=7)
    axes[0][0].legend(fontsize=7)
    save_figure(fig, "exp2_pagerank_convergence")

    df = pd.DataFrame(rows)
    df.to_csv(results_path("exp2_pagerank.csv"), index=False)
    print("\nwrote results/exp2_pagerank.csv")

    banner("PHASE 2/3 SUMMARY -- matrix-vector products")
    print(df.pivot_table(index="graph", columns="method", values="matvecs",
                         aggfunc="first").to_string())
    banner("speedup over the power iteration")
    print(df.pivot_table(index="graph", columns="method", values="speedup",
                         aggfunc="first").to_string(float_format=lambda v: f"{v:5.2f}"))
    print(f"\n  Phase 2 gate: {'PASS' if gate_ok else 'FAIL'}")
    return 0 if gate_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
