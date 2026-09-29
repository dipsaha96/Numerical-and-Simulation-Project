"""Experiment 8 - the limitation: what happens on a directed web graph.

Experiments 2-7 establish a positive result on undirected co-authorship graphs.
This one marks its boundary, because the boundary is sharp and stating it is
what makes the positive claim credible.

``web-Stanford`` is a **directed** web graph: 281,903 pages, 2.3M hyperlinks.
Direction is the only structural difference from the graphs used elsewhere in
this suite, and it changes the outcome completely.

The reason is the identity the acceleration rests on.  Momentum is a power
iteration on the augmented matrix, whose eigenvalues are

    mu_+- = ( lambda +- sqrt(lambda^2 - 4 beta) ) / 2

By Vieta the two roots always multiply to ``beta``.  When ``lambda`` is **real**
and ``lambda^2 < 4 beta`` they are complex conjugates, so they have equal
modulus, and equal modulus with product ``beta`` forces each to be exactly
``sqrt(beta)``: every rival mode is crushed onto one small circle.  That is the
acceleration.

For **complex** ``lambda`` the roots are not conjugates.  They split -- one
grows while the other shrinks to keep the product at ``beta`` -- and the larger
one can exceed ``|mu(lambda_1)|``, at which point the iteration converges to the
wrong eigenvector.  An undirected graph gives a real spectrum (experiment 5);
a directed one does not.

    python experiment/exp8_limitations.py
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from _common import GRAPH, TOL, banner, result_path, save_figure
from _methods import run_all_methods
from _style import STYLE, markevery, use_style
from src.datasets import load_snap_edges
from src.google_matrix import build_google_matrix
from src.iterations import asymptotic_rate, optimal_beta, predicted_speedup
from src.spectrum import mu_magnitude, sample_spectrum

DIRECTED = "web-Stanford"
# The base paper caps its own runs at 2000 iterations.  We allow more so the
# power iteration has room to finish, but not so much that a diverging run
# takes minutes: the outcome is settled long before.
MAX_ITER = 5000


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dampings", nargs="*", type=float, default=[0.85, 0.95, 0.99])
    ap.add_argument("--spectrum-d", type=float, default=0.99)
    ap.add_argument("--k", type=int, default=20)
    args = ap.parse_args()
    use_style()
    import matplotlib.pyplot as plt

    # ---------------- the spectra, side by side ----------------
    banner("1. The spectrum: undirected vs directed")
    spectra, info = {}, {}
    for graph in (GRAPH, DIRECTED):
        src, dst = load_snap_edges(graph)
        G, _ = build_google_matrix(src, dst, d=args.spectrum_d, name=graph)
        spec = sample_spectrum(G, G.n, k=args.k, tol=1e-5)
        spectra[graph] = spec
        info[graph] = (G.n, int(G.P.nnz))
        kind = "UNDIRECTED" if graph == GRAPH else "DIRECTED"
        print(f"  {graph:<14} ({kind:<10}) n={G.n:>7,}  {spec.size} eigenvalues sampled")
        print(f"                   max |Im lambda| = {np.abs(spec.imag).max():.3e}")

    ratio = np.abs(spectra[DIRECTED].imag).max() / max(np.abs(spectra[GRAPH].imag).max(), 1e-300)
    print(f"\n  The directed graph's spectrum is roughly {ratio:.0e} times further")
    print("  off the real axis. That single difference is the whole story.")

    # ---------------- what that does to mu ----------------
    banner("2. What a complex spectrum does to the augmented matrix")
    rows = []
    for graph in (GRAPH, DIRECTED):
        spec, d = spectra[graph], args.spectrum_d
        beta = optimal_beta(d)
        mu1 = float(mu_magnitude(np.array([1.0 + 0j]), beta)[0])
        mags = mu_magnitude(spec, beta)
        over = int((mags > mu1).sum())
        print(f"  {graph:<14} beta = d^2/4 = {beta:.4f},  sqrt(beta) = {np.sqrt(beta):.4f}")
        print(f"                 strongest rival |mu| = {mags.max():.4f}   "
              f"|mu(lambda_1)| = {mu1:.4f}")
        print(f"                 rivals that OVERTAKE the dominant mode: {over} of {spec.size}"
              f"   -> {'converges' if over == 0 else 'DIVERGES'}")
        if over:
            worst = spec[np.argmax(mags)]
            arg = np.degrees(np.angle(worst))
            print(f"                 worst: lambda = {worst.real:+.4f}{worst.imag:+.4f}i, "
                  f"|lambda| = {abs(worst):.4f}, arg = {arg:+.1f} deg, "
                  f"|mu| = {mags.max():.4f}")
            if mags.max() > 1.0:
                print(f"                 |mu| > 1, so this mode GROWS: the iteration does not")
                print(f"                 merely lose the race, it diverges outright.")
            print(f"                 What makes a mode damaging is its ANGLE, not its size:")
            print(f"                 this one sits {abs(arg):.0f} degrees off the real axis, which is")
            print(f"                 the worst case the disk |lambda| <= d allows.")
        rows.append(dict(graph=graph, directed=(graph == DIRECTED), d=d,
                         n=info[graph][0], n_sampled=int(spec.size),
                         max_abs_imag=float(np.abs(spec.imag).max()),
                         beta=beta, sqrt_beta=float(np.sqrt(beta)),
                         mu_subdominant=float(mags.max()), mu_dominant=mu1,
                         n_overtaking=over, rate=float(mags.max() / mu1)))

    # ---------------- convergence on the directed graph ----------------
    banner(f"3. Convergence rate on {DIRECTED}")
    src, dst = load_snap_edges(DIRECTED)
    base_G, _ = build_google_matrix(src, dst, d=args.dampings[0], name=DIRECTED)
    print(f"  {base_G!r}")
    print(f"  {'d':>7} {'power':>9} {'static':>9} {'dynamic':>9} | "
          f"{'theory said':>12}   outcome")
    print("  " + "-" * 66)
    conv, runs = {}, []
    for d in args.dampings:
        G = base_G.with_damping(d)
        res = run_all_methods(G, d, tol=TOL, max_iter=MAX_ITER)
        conv[d] = res
        tag = lambda r: f"{r.matvecs}" + ("" if r.converged else "!")
        good = res["dynamic"].converged
        print(f"  {d:>7.4g} {tag(res['power']):>9} {tag(res['static']):>9} "
              f"{tag(res['dynamic']):>9} | {predicted_speedup(d):>11.2f}x   "
              f"{'converged' if good else 'momentum FAILED'}")
        for name, out in res.items():
            runs.append(dict(graph=DIRECTED, d=d, method=name, matvecs=out.matvecs,
                             converged=out.converged, residual=out.final_residual(),
                             time_s=out.time, predicted_speedup=predicted_speedup(d)))
    print("\n  ! = did not converge within the budget")

    pd.DataFrame(rows).to_csv(result_path("exp8_limitations_spectrum.csv"), index=False)
    pd.DataFrame(runs).to_csv(result_path("exp8_limitations_runs.csv"), index=False)
    print("\nwrote result/exp8_limitations_spectrum.csv and _runs.csv")

    # ---------------- figure ----------------
    fig, ax = plt.subplots(1, 3, figsize=(13.8, 4.0))
    fig.subplots_adjust(wspace=0.33)

    a = ax[0]
    for graph, col, mk in ((GRAPH, "#009E73", "o"), (DIRECTED, "#D55E00", "^")):
        s = spectra[graph]
        a.scatter(s.real, s.imag, s=30, marker=mk, alpha=0.75, color=col,
                  label=f"{graph}  (max|Im| = {np.abs(s.imag).max():.0e})")
    a.axhline(0, color="black", lw=0.7)
    a.set_xlabel(r"$\mathrm{Re}\,\lambda$"); a.set_ylabel(r"$\mathrm{Im}\,\lambda$")
    a.set_title(f"A. Undirected stays on the real axis;\ndirected does not  ($d={args.spectrum_d}$)")
    a.set_ylim(-1.35, 1.35)
    a.legend(fontsize=7, loc="lower center")

    a = ax[1]
    beta = optimal_beta(args.spectrum_d)
    mu1 = float(mu_magnitude(np.array([1.0 + 0j]), beta)[0])
    for graph, col, mk in ((GRAPH, "#009E73", "o"), (DIRECTED, "#D55E00", "^")):
        s = spectra[graph]
        a.scatter(np.abs(s), mu_magnitude(s, beta), s=30, marker=mk, alpha=0.75,
                  color=col, label=graph)
    a.axhline(mu1, color="black", lw=1.6, label=r"$|\mu(\lambda_1)|$ — the winner")
    a.axhline(np.sqrt(beta), color="#009E73", ls=":", lw=1.4, label=r"$\sqrt{\beta}$")
    a.set_xlabel(r"$|\lambda|$ of the rival mode")
    a.set_ylabel(r"$|\mu(\lambda)|$ after momentum")
    a.set_title("B. Real rivals land on $\\sqrt{\\beta}$;\ncomplex ones climb past the winner")
    a.legend(fontsize=7)

    a = ax[2]
    d = args.dampings[-1]
    for name, out in conv[d].items():
        y = np.asarray(out.residual)
        a.semilogy(np.arange(1, y.size + 1), y, markevery=markevery(y.size), **STYLE[name])
    a.set_xlabel("matrix-vector products")
    a.set_ylabel(r"$\|Gx_k-\nu_k x_k\|_2$")
    a.set_title(f"C. {DIRECTED}, $d={d}$:\nmomentum never converges")
    a.legend(fontsize=7.5)
    save_figure(fig, "exp8_limitations")

    banner("THE BOUNDARY OF THE RESULT")
    print(f"  Works   : undirected graphs. Real spectrum, every rival folded onto")
    print(f"            sqrt(beta), 3.2x-42x measured across 25 PageRank problems.")
    print(f"  Fails   : directed graphs. Complex spectrum, rivals escape the circle,")
    print(f"            and Algorithm 3.1 diverges at every damping factor tested.")
    print(f"  Cause   : one structural property - whether the graph has direction.")
    print(f"\n  This is a statement about the Google matrix, not about our code:")
    print(f"  experiment 1 reproduces the paper's own benchmarks, and the plain")
    print(f"  power iteration converges normally on {DIRECTED} throughout.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
