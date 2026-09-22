"""Phase 5: when is the *ranking* right, as opposed to the residual small?

The base paper monitors ``||A x - nu x||_2`` and stops at ``1e-12``.  That is the
correct stopping rule for an eigensolver but not for a ranking application: a
search engine needs the order of the top results, is indifferent to the fourth
significant figure of each score, and is entirely indifferent to the tail.

This script scores every iterate of every method against a high-accuracy
reference and reports how many matrix-vector products each one needs to reach a
*ranking* criterion -- ``precision@100 = 1.0`` and Kendall tau above 0.99 --
rather than an eigen-residual criterion.  The gap between the two is the
practically interesting number, and it is the part of the study that the base
paper's abstract benchmarks cannot produce at all.

Usage::

    python experiments/exp4_ranking.py --graph cit-HepPh --dampings 0.85 0.99
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from _common import banner, results_path
from src.datasets import load_snap_edges, synthetic_web_graph
from src.google_matrix import build_google_matrix, normalise_ranking
from src.iterations import power_iteration
from src.metrics import iterations_to_threshold, kendall_tau, l1_error, precision_at_k
from src.pagerank_methods import run_methods
from src.plotting import METHOD_STYLE, markevery, save_figure, use_style

METHODS = ["power", "static_d", "dynamic_safe", "dynamic_guard"]
TOL = 1e-12
MAX_ITER = 40_000
TRACK_LIMIT = 4000          # cap on retained iterates, to bound memory


def reference_ranking(G) -> np.ndarray:
    """High-accuracy PageRank vector used to score every partial ranking."""
    G.reset()
    return normalise_ranking(power_iteration(G, tol=1e-14, max_iter=100_000).x)


def score_run(res, reference: np.ndarray, k: int, stride: int) -> pd.DataFrame:
    """Per-iterate ranking quality for one solver run."""
    rows = []
    for i, x in enumerate(res.iterates):
        if i % stride and i != len(res.iterates) - 1:
            continue
        rows.append(dict(
            step=i + 1,
            residual=res.residual[i] if i < len(res.residual) else np.nan,
            l1_error=l1_error(x, reference),
            precision_at_k=precision_at_k(x, reference, k),
            kendall=kendall_tau(x, reference, k),
        ))
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", default="cit-HepPh")
    parser.add_argument("--dampings", nargs="*", type=float, default=[0.85, 0.99])
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    use_style()
    import matplotlib.pyplot as plt

    try:
        if args.offline:
            raise RuntimeError("offline mode")
        src, dst = load_snap_edges(args.graph)
    except Exception as exc:                        # noqa: BLE001
        print(f"{args.graph} unavailable ({exc}); using a synthetic graph")
        src, dst = synthetic_web_graph(n=20_000, seed=1)

    base, _ = build_google_matrix(src, dst, d=args.dampings[0], name=args.graph)
    rows, curves = [], []

    for d in args.dampings:
        banner(f"{args.graph}, d = {d}: residual convergence vs. ranking convergence")
        G = base.with_damping(d)
        reference = reference_ranking(G)

        results = run_methods(G, d, tol=TOL, max_iter=MAX_ITER,
                              methods=METHODS, track_iterates=True)
        for name, res in results.items():
            stride = max(1, len(res.iterates) // TRACK_LIMIT)
            curve = score_run(res, reference, args.k, stride)
            curves.append((d, name, curve))

            mv_rank = iterations_to_threshold(curve["precision_at_k"], 1.0, rising=True)
            mv_tau = iterations_to_threshold(curve["kendall"], 0.99, rising=True)
            mv_rank = int(curve["step"].iloc[mv_rank]) if mv_rank is not None else None
            mv_tau = int(curve["step"].iloc[mv_tau]) if mv_tau is not None else None
            ratio = (res.matvecs / mv_rank) if mv_rank else float("nan")

            print(f"  {name:>14s}: residual 1e-12 at {res.matvecs:6d} mv | "
                  f"precision@{args.k}=1.0 at {str(mv_rank):>6s} mv | "
                  f"tau>0.99 at {str(mv_tau):>6s} mv | "
                  f"ranking ready {ratio:5.1f}x sooner")
            rows.append(dict(graph=args.graph, d=d, method=name, k=args.k,
                             matvecs_residual=res.matvecs,
                             matvecs_precision=mv_rank, matvecs_kendall=mv_tau,
                             ranking_speedup=ratio, converged=res.converged))

    df = pd.DataFrame(rows)
    df.to_csv(results_path("exp4_ranking.csv"), index=False)
    pd.concat([c.assign(d=d, method=m) for d, m, c in curves]).to_csv(
        results_path("exp4_ranking_curves.csv"), index=False)
    print("\nwrote results/exp4_ranking.csv and results/exp4_ranking_curves.csv")

    fig, axes = plt.subplots(len(args.dampings), 2,
                             figsize=(10, 3.4 * len(args.dampings)), squeeze=False)
    for row, d in enumerate(args.dampings):
        ax = axes[row][0]
        for dd, name, curve in curves:
            if dd != d:
                continue
            ax.semilogy(curve["step"], curve["residual"],
                        markevery=markevery(len(curve)), **METHOD_STYLE[name])
        ax.set_xlabel("matrix-vector products")
        ax.set_ylabel(r"$\|Gx_k-\nu_k x_k\|_2$")
        ax.set_title(f"{args.graph}, $d={d}$: eigen-residual")

        ax = axes[row][1]
        for dd, name, curve in curves:
            if dd != d:
                continue
            ax.plot(curve["step"], curve["precision_at_k"],
                    markevery=markevery(len(curve)), **METHOD_STYLE[name])
        ax.axhline(1.0, color="black", ls="--", lw=1.0)
        ax.set_xlabel("matrix-vector products")
        ax.set_ylabel(f"precision@{args.k}")
        ax.set_ylim(0, 1.05)
        ax.set_title("top-100 ranking quality")
    axes[0][0].legend(fontsize=7)
    save_figure(fig, "exp4_ranking_quality")

    banner("RESIDUAL CONVERGENCE vs RANKING CONVERGENCE")
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
