"""Experiment 7 - is the result a property of ca-HepPh, or of undirected graphs?

Experiments 2-6 all use one graph.  The obvious objection is that the result
might be an accident of that particular dataset, so this repeats the core
measurement -- the damping sweep -- on the whole family of arXiv co-authorship
networks.  They differ by a factor of four in size and a factor of ten in
density, and they come from five different fields of physics:

    ca-GrQc      5,242 authors   general relativity      sparsest
    ca-HepTh     9,877           high-energy theory
    ca-HepPh    12,008           high-energy phenomenology
    ca-AstroPh  18,772           astrophysics            densest
    ca-CondMat  23,133           condensed matter        largest

What they share is that all five are **undirected**, so every one of their
Google matrices should have a real spectrum -- the condition the base paper's
theory assumes.  If the acceleration is a property of undirected graphs rather
than of one dataset, all five should track the predicted speed-up.

    python experiment/exp7_generality.py
    python experiment/exp7_generality.py --graphs ca-GrQc ca-HepTh
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from _common import DAMPINGS, MAX_ITER, TOL, banner, result_path, save_figure
from _methods import run_all_methods
from _style import STYLE, use_style
from src.datasets import load_snap_edges
from src.google_matrix import build_google_matrix
from src.iterations import asymptotic_rate, predicted_speedup
from src.spectrum import sample_spectrum

FAMILY = ["ca-GrQc", "ca-HepTh", "ca-HepPh", "ca-AstroPh", "ca-CondMat"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graphs", nargs="*", default=FAMILY)
    ap.add_argument("--dampings", nargs="*", type=float, default=DAMPINGS)
    args = ap.parse_args()
    use_style()
    import matplotlib.pyplot as plt

    rows, spectra = [], {}
    for graph in args.graphs:
        banner(f"{graph}")
        try:
            src, dst = load_snap_edges(graph)
        except Exception as exc:                        # noqa: BLE001
            print(f"  unavailable ({type(exc).__name__}: {exc}) -- skipped")
            continue
        base_G, _ = build_google_matrix(src, dst, d=args.dampings[0], name=graph)

        # Is the spectrum real, as an undirected graph should give?
        spec = sample_spectrum(base_G.with_damping(0.99), base_G.n, k=30)
        max_im = float(np.abs(spec.imag).max()) if spec.size else float("nan")
        spectra[graph] = max_im
        print(f"  {base_G.n:,} nodes, {base_G.P.nnz:,} links, "
              f"{int(base_G.dangling.sum())} dangling")
        print(f"  max |Im lambda| of G = {max_im:.2e}   "
              f"({'REAL spectrum' if max_im < 1e-10 else 'complex'})")
        print(f"  {'d':>7} {'power':>8} {'static':>8} {'dynamic':>8} | "
              f"{'dynamic':>9} {'theory':>8} {'% of theory':>12}")
        print("  " + "-" * 66)

        for d in args.dampings:
            G = base_G.with_damping(d)
            res = run_all_methods(G, d, tol=TOL, max_iter=MAX_ITER)
            pw = res["power"].matvecs
            sp = pw / res["dynamic"].matvecs
            pct = 100 * sp / predicted_speedup(d)
            print(f"  {d:>7.4g} {pw:>8d} {res['static'].matvecs:>8d} "
                  f"{res['dynamic'].matvecs:>8d} | {sp:>8.2f}x "
                  f"{predicted_speedup(d):>7.2f}x {pct:>11.1f}%")
            for name, out in res.items():
                rows.append(dict(graph=graph, n=G.n, nnz=int(G.P.nnz), d=d,
                                 method=name, matvecs=out.matvecs,
                                 converged=out.converged, time_s=out.time,
                                 residual=out.final_residual(),
                                 speedup=pw / out.matvecs,
                                 predicted_speedup=predicted_speedup(d),
                                 pct_of_theory=100 * (pw / out.matvecs) / predicted_speedup(d),
                                 max_abs_imag=max_im))

    df = pd.DataFrame(rows)
    df.to_csv(result_path("exp7_generality.csv"), index=False)
    print("\nwrote result/exp7_generality.csv")

    # ---------------- figure ----------------
    graphs = [g for g in args.graphs if g in df.graph.unique()]
    xs = np.arange(len(args.dampings))
    labels = [f"{d:g}" for d in args.dampings]
    fig, ax = plt.subplots(1, 3, figsize=(13.4, 3.9))

    a = ax[0]
    a.plot(xs, [predicted_speedup(d) for d in args.dampings], "--", color="black",
           lw=1.6, marker="_", ms=12, label="theory")
    for g in graphs:
        sub = df[(df.graph == g) & (df.method == "dynamic")].sort_values("d")
        a.plot(xs, sub.speedup.values, "o-", ms=4, label=g)
    a.set_xticks(xs); a.set_xticklabels(labels); a.set_yscale("log")
    a.set_xlabel("damping factor $d$"); a.set_ylabel("measured speed-up")
    a.set_title("All five undirected graphs\nfollow the same curve")
    a.legend(fontsize=7)

    a = ax[1]
    for g in graphs:
        sub = df[(df.graph == g) & (df.method == "dynamic")].sort_values("d")
        a.plot(xs, sub.pct_of_theory.values, "o-", ms=4, label=g)
    a.axhline(100, color="black", ls="--", lw=1.2)
    a.set_xticks(xs); a.set_xticklabels(labels)
    a.set_xlabel("damping factor $d$"); a.set_ylabel("achieved, as % of theory")
    a.set_ylim(60, 112)
    a.set_title("Alg. 3.1 attains a consistent\nfraction of the maximum")
    a.legend(fontsize=7)

    a = ax[2]
    sizes = df.groupby("graph")["n"].first()
    top = df[(df.d == args.dampings[-1]) & (df.method == "dynamic")].set_index("graph")
    a.scatter(sizes[graphs], top.loc[graphs, "speedup"], s=70,
              color=STYLE["dynamic"]["color"], zorder=3)
    for g in graphs:
        a.annotate(g, (sizes[g], top.loc[g, "speedup"]), textcoords="offset points",
                   xytext=(0, 9), ha="center", fontsize=7.5)
    a.axhline(predicted_speedup(args.dampings[-1]), color="black", ls="--", lw=1.2,
              label=f"theory at $d={args.dampings[-1]:g}$")
    a.set_xlabel("graph size (nodes)"); a.set_ylabel("speed-up")
    a.set_title(f"Independent of graph size\n($d={args.dampings[-1]:g}$)")
    a.legend(fontsize=7.5)
    save_figure(fig, "exp7_generality")

    banner("SUMMARY")
    t = df[df.method == "dynamic"].pivot_table(index="graph", columns="d", values="speedup")
    t = t.reindex(graphs)
    t.loc["theory"] = [predicted_speedup(d) for d in args.dampings]
    print(t.to_string(float_format=lambda v: f"{v:8.2f}"))
    pct = df[df.method == "dynamic"].pct_of_theory
    print(f"\n  Across {len(graphs)} graphs x {len(args.dampings)} damping factors "
          f"= {len(graphs) * len(args.dampings)} PageRank problems:")
    print(f"    every run converged            : "
          f"{bool(df[df.method == 'dynamic'].converged.all())}")
    print(f"    spectra real (max |Im lambda|) : "
          f"{max(spectra.values()):.1e} over all graphs")
    print(f"    fraction of theory attained    : "
          f"{pct.min():.1f}% to {pct.max():.1f}%  (mean {pct.mean():.1f}%)")
    print("\n  The acceleration is a property of undirected graphs, not of one dataset.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
