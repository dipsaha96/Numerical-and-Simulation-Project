"""Phase 4: the damping-factor sweep -- the headline experiment.

For the Google matrix the spectral ratio is exactly the damping factor,
``r = |lambda_2/lambda_1| = d``.  Substituting into the paper's accelerated rate
``rho(r) = r/(1+sqrt(1-r^2))`` gives a *parameter-free prediction* of how much
faster the momentum methods should be:

    d      rho(d)    predicted speedup = log(rho)/log(d)
    0.85    0.557     3.60x
    0.90    0.627     4.43x
    0.95    0.724     6.30x
    0.99    0.868     14.13x
    0.999   0.956     44.72x

This script measures the real speedup at each ``d`` and plots it against that
curve.  The regime that matters most for the project is exactly the regime where
plain power iteration is worst: as ``d -> 1`` the spectral gap closes, the power
iteration crawls, and the acceleration grows without bound.

Usage::

    python experiments/exp3_damping.py --graphs wiki-Vote cit-HepPh
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from _common import banner, results_path
from src.datasets import load_snap_edges, synthetic_web_graph
from src.google_matrix import build_google_matrix, normalise_ranking
from src.iterations import asymptotic_rate, optimal_beta, predicted_speedup
from src.pagerank_methods import run_methods
from src.metrics import l1_error
from src.plotting import METHOD_STYLE, save_figure, use_style

DEFAULT_DAMPINGS = [0.85, 0.90, 0.95, 0.99, 0.999]
DEFAULT_GRAPHS = ["wiki-Vote", "cit-HepPh"]
TOL = 1e-12
MAX_ITER = 80_000          # d = 0.999 needs far more than the paper's 2000


def sweep_graph(key: str, src, dst, dampings, rows: list[dict]) -> None:
    # Build the link matrix once; `with_damping` re-uses the sparse structure,
    # which matters on the larger graphs where construction dominates.
    base, _ = build_google_matrix(src, dst, d=dampings[0], name=key)
    print(f"  {base!r}")

    for d in dampings:
        G = base.with_damping(d)
        beta_opt = optimal_beta(d)
        results = run_methods(G, d, tol=TOL, max_iter=MAX_ITER)
        power = results["power"]
        reference = normalise_ranking(power.x)
        predicted = predicted_speedup(d)

        # A run that stopped early has no meaningful cost, so it has no
        # meaningful speedup either: reporting `power.matvecs / r.matvecs` for a
        # run that gave up after 1004 of a possible 80,000 matrix-vector
        # products would read as a 19x win for a method that never converged.
        def _cell(n, r):
            if not r.converged:
                return f"{n}={r.matvecs}{'!' if r.diverged else '*'}(--)"
            return f"{n}={r.matvecs}({power.matvecs / r.matvecs:.2f}x)"
        summary = "  ".join(_cell(n, r) for n, r in results.items() if n != "power")
        print(f"    d={d:<6}: power={power.matvecs:<6d} {summary} | theory {predicted:.2f}x")
        if not power.converged:
            print(f"             (power iteration hit the {MAX_ITER} matvec cap)")

        for name, res in results.items():
            rows.append(dict(
                graph=key, n=G.n, d=d, method=name,
                matvecs=res.matvecs, iterations=res.n_iter, time_s=res.time,
                converged=res.converged, diverged=res.diverged,
                residual=res.final_residual(),
                speedup=(power.matvecs / res.matvecs) if res.converged else float("nan"),
                predicted_speedup=predicted,
                theoretical_rate=d if name == "power" else asymptotic_rate(d),
                observed_rate=_empirical_rate(res.residual),
                l1_agreement=l1_error(res.x, reference),
                safeguard_hits=res.safeguard_hits, backtracks=res.backtracks,
                beta_opt=beta_opt,
                beta_final=float(np.median(res.beta[-20:])) if res.beta else 0.0,
            ))


def _empirical_rate(residual: list[float], tail: int = 50) -> float:
    """Geometric convergence rate fitted to the last ``tail`` residuals.

    The asymptotic rate is what the theory predicts, so it must be measured
    away from the pre-asymptotic regime -- hence fitting only the tail.
    """
    y = np.asarray(residual, dtype=float)
    y = y[y > 0]
    if y.size < 5:
        return float("nan")
    y = y[-min(tail, y.size):]
    slope = np.polyfit(np.arange(y.size), np.log(y), 1)[0]
    return float(np.exp(slope))


def plot_sweep(df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    graphs = df["graph"].unique()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))

    # ---- left: measured vs predicted speedup ----
    ax = axes[0]
    dd = np.linspace(0.80, 0.9995, 400)
    ax.plot(dd, [predicted_speedup(v) for v in dd], color="black", ls="--", lw=1.4,
            label=r"theory: $\log\rho(d)/\log d$")
    for i, g in enumerate(graphs):
        for method in ("static_d", "dynamic_safe"):
            sub = df[(df.graph == g) & (df.method == method)].sort_values("d")
            style = dict(METHOD_STYLE[method])
            style["label"] = f"{g} -- {style['label']}"
            ax.plot(sub["d"], sub["speedup"], ls="-" if i == 0 else ":", **style)
    ax.set_xlabel("damping factor $d$")
    ax.set_ylabel(r"speedup over power iteration")
    ax.axhline(1.0, color="red", ls=":", lw=1.2, label="parity with power iteration")
    ax.set_yscale("log")
    ax.set_title("Measured acceleration follows the theory")
    ax.legend(fontsize=7)

    # ---- right: matvecs to convergence ----
    ax = axes[1]
    for i, g in enumerate(graphs):
        for method in ("power", "static_d", "dynamic_safe"):
            sub = df[(df.graph == g) & (df.method == method)].sort_values("d")
            style = dict(METHOD_STYLE[method])
            style["label"] = f"{g} -- {style['label']}" if i == 0 else None
            ax.plot(sub["d"], sub["matvecs"], ls="-" if i == 0 else ":", **style)
    ax.set_xlabel("damping factor $d$")
    ax.set_ylabel("matrix-vector products to $10^{-12}$")
    ax.set_yscale("log")
    ax.set_title("Cost of ranking as the spectral gap closes")
    ax.legend(fontsize=7)

    save_figure(fig, "exp3_damping_sweep")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", nargs="*", default=DEFAULT_GRAPHS)
    parser.add_argument("--dampings", nargs="*", type=float, default=DEFAULT_DAMPINGS)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    use_style()
    rows: list[dict] = []

    banner("Theoretical prediction (r = d for the Google matrix)")
    print(f"  {'d':>7s}  {'rho(d)':>8s}  {'predicted speedup':>18s}")
    for d in args.dampings:
        print(f"  {d:>7.4g}  {asymptotic_rate(d):>8.4f}  {predicted_speedup(d):>17.2f}x")

    for key in args.graphs:
        banner(f"Damping sweep: {key}")
        try:
            if args.offline:
                raise RuntimeError("offline mode")
            src, dst = load_snap_edges(key)
        except Exception as exc:                        # noqa: BLE001
            print(f"  {key} unavailable ({exc}); using a synthetic graph")
            src, dst = synthetic_web_graph(n=20_000, seed=abs(hash(key)) % 2**31)
        sweep_graph(key, src, dst, args.dampings, rows)

    df = pd.DataFrame(rows)
    path = results_path("exp3_damping.csv")
    df.to_csv(path, index=False)
    print(f"\nwrote results/{path.name}   (! = diverged, * = hit the iteration cap)")
    plot_sweep(df)

    banner("MEASURED vs PREDICTED SPEEDUP")
    table = df[df.method != "power"].pivot_table(
        index=["graph", "d"], columns="method", values="speedup")
    table["theory"] = df[df.method == "dynamic_safe"].set_index(["graph", "d"])["predicted_speedup"]
    print(table.to_string(float_format=lambda v: f"{v:6.2f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
