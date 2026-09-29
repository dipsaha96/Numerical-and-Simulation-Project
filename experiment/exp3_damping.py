"""Experiment 3 - the damping sweep.  This is the headline result.

For a Google matrix the spectral ratio is exactly the damping factor,
``r = |lambda_2/lambda_1| = d``, because ``lambda_1 = 1`` by stochasticity and
``|lambda_2| = d`` (Haveliwala & Kamvar 2003, confirmed by measurement in
experiment 2).  Substituting into the base paper's rate
``rho(r) = r/(1+sqrt(1-r^2))`` therefore gives a *parameter-free prediction* of
the speed-up at every damping factor, with nothing fitted:

    d       rho(d)    predicted speed-up
    0.85    0.5567     3.60x
    0.90    0.6268     4.43x
    0.95    0.7239     6.30x
    0.99    0.8676    14.13x
    0.999   0.9562    44.72x

This script measures the real thing and plots it against that curve.  The
regime that matters is exactly the one where plain power iteration is worst: as
``d -> 1`` the spectral gap closes, the power iteration crawls, and the
acceleration grows without bound.

    python experiment/exp3_damping.py
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from _common import DAMPINGS, GRAPH, MAX_ITER, TOL, banner, result_path, save_figure
from _methods import run_all_methods
from _style import STYLE, use_style
from src.datasets import load_snap_edges
from src.google_matrix import build_google_matrix, normalise_ranking
from src.iterations import asymptotic_rate, optimal_beta, predicted_speedup
from src.metrics import l1_error


def empirical_rate(residual, tail: int = 40) -> float:
    """Geometric rate fitted to the last few residuals (the asymptotic regime)."""
    y = np.asarray(residual, dtype=float)
    y = y[y > 0]
    if y.size < 5:
        return float("nan")
    y = y[-min(tail, y.size):]
    return float(np.exp(np.polyfit(np.arange(y.size), np.log(y), 1)[0]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dampings", nargs="*", type=float, default=DAMPINGS)
    args = ap.parse_args()
    use_style()
    import matplotlib.pyplot as plt

    banner("Theoretical prediction  (r = d for the Google matrix)")
    print(f"  {'d':>7}  {'rho(d)':>8}  {'predicted speed-up':>19}")
    for d in args.dampings:
        print(f"  {d:>7.4g}  {asymptotic_rate(d):>8.4f}  {predicted_speedup(d):>18.2f}x")

    src, dst = load_snap_edges(GRAPH)
    base_G, _ = build_google_matrix(src, dst, d=args.dampings[0], name=GRAPH)
    banner(f"Measured on {GRAPH}  ({base_G.n:,} nodes, {base_G.P.nnz:,} links)")
    print(f"  {'d':>7} {'power':>8} {'static':>8} {'dynamic':>8} | "
          f"{'static':>8} {'dynamic':>8} | {'theory':>8}")
    print("  " + "-" * 68)

    rows = []
    for d in args.dampings:
        G = base_G.with_damping(d)
        res = run_all_methods(G, d, tol=TOL, max_iter=MAX_ITER)
        pw = res["power"].matvecs
        reference = normalise_ranking(res["power"].x)
        print(f"  {d:>7.4g} {pw:>8d} {res['static'].matvecs:>8d} "
              f"{res['dynamic'].matvecs:>8d} | "
              f"{pw / res['static'].matvecs:>7.2f}x {pw / res['dynamic'].matvecs:>7.2f}x | "
              f"{predicted_speedup(d):>7.2f}x")
        for name, out in res.items():
            rows.append(dict(graph=GRAPH, n=G.n, d=d, method=name,
                             matvecs=out.matvecs, iterations=out.n_iter,
                             time_s=out.time, converged=out.converged,
                             residual=out.final_residual(),
                             speedup=pw / out.matvecs,
                             predicted_speedup=predicted_speedup(d),
                             theoretical_rate=d if name == "power" else asymptotic_rate(d),
                             observed_rate=empirical_rate(out.residual),
                             l1_agreement=l1_error(out.x, reference),
                             beta_opt=optimal_beta(d),
                             beta_final=float(np.median(out.beta[-20:])) if out.beta else 0.0))

    df = pd.DataFrame(rows)
    df.to_csv(result_path("exp3_damping.csv"), index=False)
    print("\nwrote result/exp3_damping.csv")

    # ---------------- figure ----------------
    fig, ax = plt.subplots(1, 3, figsize=(13.4, 3.9))
    # The damping values cluster near 1 (0.99 and 0.999 are visually on top of
    # each other on a linear axis), so plot them at evenly spaced positions and
    # label the ticks with the actual values.
    xs = np.arange(len(args.dampings))
    labels = [f"{d:g}" for d in args.dampings]

    a = ax[0]
    theory = [predicted_speedup(d) for d in args.dampings]
    a.plot(xs, theory, "--", color="black", lw=1.5, marker="_", ms=10,
           label=r"theory: $\log\rho(d)/\log d$")
    for name in ("static", "dynamic"):
        sub = df[df.method == name].sort_values("d")
        a.plot(xs, sub["speedup"].values, **STYLE[name])
    a.set_xticks(xs); a.set_xticklabels(labels)
    a.set_xlabel("damping factor $d$"); a.set_ylabel("measured speed-up")
    a.set_yscale("log"); a.set_title("Measured acceleration follows the theory")
    a.legend(fontsize=7.5)

    a = ax[1]
    for name in ("power", "static", "dynamic"):
        sub = df[df.method == name].sort_values("d")
        a.plot(xs, sub["matvecs"].values, **STYLE[name])
    a.set_xticks(xs); a.set_xticklabels(labels)
    a.set_xlabel("damping factor $d$")
    a.set_ylabel(r"matrix-vector products to $10^{-12}$")
    a.set_yscale("log"); a.set_title("Cost as the spectral gap closes")
    a.legend(fontsize=7.5)

    a = ax[2]
    sub = df[df.method == "dynamic"].sort_values("d")
    pct = (100 * sub["speedup"] / sub["predicted_speedup"]).values
    a.plot(xs, pct, "o-", color=STYLE["dynamic"]["color"])
    a.axhline(100, color="black", ls="--", lw=1.2, label="theoretical maximum")
    for x, y in zip(xs, pct):
        a.annotate(f"{y:.0f}%", (x, y), textcoords="offset points", xytext=(0, 8),
                   ha="center", fontsize=8)
    a.set_xticks(xs); a.set_xticklabels(labels)
    a.set_xlabel("damping factor $d$")
    a.set_ylabel("achieved, as % of theory")
    a.set_ylim(60, 112)
    a.set_title("Alg. 3.1 attains 89-94% of the\ntheoretical maximum")
    a.legend(fontsize=7.5, loc="lower right")
    save_figure(fig, "exp3_damping")

    banner("MEASURED vs PREDICTED")
    t = df[df.method != "power"].pivot_table(index="d", columns="method", values="speedup")
    t["theory"] = df[df.method == "dynamic"].set_index("d")["predicted_speedup"]
    t["dynamic % of theory"] = 100 * t["dynamic"] / t["theory"]
    print(t.to_string(float_format=lambda v: f"{v:8.2f}"))
    print("\n  Observed asymptotic rates against the predicted rho(d):")
    for d in args.dampings:
        got = df[(df.d == d) & (df.method == "dynamic")]["observed_rate"].iloc[0]
        print(f"    d={d:<6}: measured {got:.4f}   predicted {asymptotic_rate(d):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
