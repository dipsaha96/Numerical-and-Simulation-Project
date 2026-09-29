"""Experiment 4 - when is the *ranking* right, as opposed to the residual small?

The base paper stops when the eigen-residual ``||Ax - nu x||`` falls below a
tolerance.  That is the right rule for an eigensolver, but a ranking
application cares about something else: the *order* of the top results.  It is
indifferent to the fourth significant figure of each score and indifferent
entirely to the tail.

This script scores every iterate against a high-accuracy reference and reports
how many matrix-vector products each method needs to reach a *ranking*
criterion -- precision@100 = 1.0, and Kendall tau above 0.99 -- rather than a
residual criterion.  The gap between the two is the practically interesting
number, and it is something the base paper's abstract benchmarks cannot
produce at all.

    python experiment/exp4_ranking.py
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
from src.iterations import power_iteration
from src.metrics import iterations_to_threshold, kendall_tau, l1_error, precision_at_k

TRACK_LIMIT = 4000       # cap on retained iterates, to bound memory


def score_run(res, reference, k, stride):
    rows = []
    for i, x in enumerate(res.iterates):
        if i % stride and i != len(res.iterates) - 1:
            continue
        rows.append(dict(step=i + 1,
                         residual=res.residual[i] if i < len(res.residual) else np.nan,
                         l1_error=l1_error(x, reference),
                         precision_at_k=precision_at_k(x, reference, k),
                         kendall=kendall_tau(x, reference, k)))
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dampings", nargs="*", type=float, default=[0.85, 0.99])
    ap.add_argument("--k", type=int, default=100)
    args = ap.parse_args()
    use_style()
    import matplotlib.pyplot as plt

    src, dst = load_snap_edges(GRAPH)
    base_G, _ = build_google_matrix(src, dst, d=args.dampings[0], name=GRAPH)

    rows, curves = [], []
    for d in args.dampings:
        banner(f"{GRAPH}, d = {d}: residual convergence vs ranking convergence")
        G = base_G.with_damping(d)
        G.reset()
        reference = normalise_ranking(power_iteration(G, tol=1e-14, max_iter=200_000).x)

        res = run_all_methods(G, d, tol=TOL, max_iter=MAX_ITER, track_iterates=True)
        for name, out in res.items():
            stride = max(1, len(out.iterates) // TRACK_LIMIT)
            curve = score_run(out, reference, args.k, stride)
            curves.append((d, name, curve))

            i_p = iterations_to_threshold(curve["precision_at_k"], 1.0, rising=True)
            i_t = iterations_to_threshold(curve["kendall"], 0.99, rising=True)
            mv_p = int(curve["step"].iloc[i_p]) if i_p is not None else None
            mv_t = int(curve["step"].iloc[i_t]) if i_t is not None else None
            ratio = (out.matvecs / mv_p) if mv_p else float("nan")
            print(f"  {name:>8s}: residual 1e-12 at {out.matvecs:6d} mv | "
                  f"precision@{args.k}=1.0 at {str(mv_p):>5s} mv | "
                  f"tau>0.99 at {str(mv_t):>5s} mv | ranking ready {ratio:5.1f}x sooner")
            rows.append(dict(graph=GRAPH, d=d, method=name, k=args.k,
                             matvecs_residual=out.matvecs, matvecs_precision=mv_p,
                             matvecs_kendall=mv_t, ranking_speedup=ratio,
                             converged=out.converged))

    df = pd.DataFrame(rows)
    df.to_csv(result_path("exp4_ranking.csv"), index=False)
    pd.concat([c.assign(d=d, method=m) for d, m, c in curves]).to_csv(
        result_path("exp4_ranking_curves.csv"), index=False)
    print("\nwrote result/exp4_ranking.csv and result/exp4_ranking_curves.csv")

    fig, axes = plt.subplots(len(args.dampings), 2,
                             figsize=(10, 3.4 * len(args.dampings)), squeeze=False)
    for row, d in enumerate(args.dampings):
        sel = [(n, c) for dd, n, c in curves if dd == d]
        a = axes[row][0]
        for name, c in sel:
            a.semilogy(c["step"], c["residual"], markevery=markevery(len(c)), **STYLE[name])
        a.set_xlabel("matrix-vector products")
        a.set_ylabel(r"$\|Gx_k-\nu_k x_k\|_2$")
        a.set_title(f"{GRAPH}, $d={d}$: eigen-residual")

        a = axes[row][1]
        for name, c in sel:
            a.plot(c["step"], c["precision_at_k"], markevery=markevery(len(c)), **STYLE[name])
        a.axhline(1.0, color="black", ls="--", lw=1.0)
        a.set_xlabel("matrix-vector products")
        a.set_ylabel(f"precision@{args.k}")
        a.set_ylim(0, 1.05)
        a.set_title(f"$d={d}$: top-{args.k} ranking quality")
    axes[0][0].legend(fontsize=7.5)
    save_figure(fig, "exp4_ranking")

    banner("RESIDUAL CONVERGENCE vs RANKING CONVERGENCE")
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
