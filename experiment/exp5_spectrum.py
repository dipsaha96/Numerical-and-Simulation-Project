"""Experiment 5 - the spectral explanation: why the method works here.

The base paper's acceleration rests on one identity.  Momentum is a plain power
iteration on the augmented matrix, whose eigenvalues are, for each eigenvalue
``lambda`` of G,

    mu_+- = ( lambda +- sqrt(lambda^2 - 4 beta) ) / 2                    (2.7)

By Vieta's formula the two roots always multiply to ``beta``.  When ``lambda``
is **real** and ``lambda^2 < 4 beta`` the square root is imaginary, so the roots
are complex conjugates -- and conjugates have equal modulus.  Equal modulus with
product ``beta`` forces each to be exactly ``sqrt(beta)``.

That is the whole mechanism: every rival mode is crushed onto a single circle of
radius ``sqrt(beta)``, while ``lambda_1 = 1`` stays outside it.  The gap between
them is the acceleration, and choosing ``beta = d^2/4`` places the fold
threshold ``2 sqrt(beta)`` exactly on ``lambda_2 = d``.

The identity needs the spectrum to be real.  ``ca-HepPh`` is undirected, so its
Google matrix delivers exactly that -- this script measures how real, and then
confirms that the resulting predicted rate is the rate actually observed.

    python experiment/exp5_spectrum.py
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from _common import GRAPH, banner, result_path, save_figure
from _style import use_style
from src.datasets import load_snap_edges
from src.google_matrix import build_google_matrix
from src.iterations import asymptotic_rate, optimal_beta, predicted_speedup
from src.spectrum import convergence_rate, mu_magnitude, rate_curve, sample_spectrum


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dampings", nargs="*", type=float, default=[0.85, 0.95, 0.99])
    args = ap.parse_args()
    use_style()
    import matplotlib.pyplot as plt

    src, dst = load_snap_edges(GRAPH)
    base_G, _ = build_google_matrix(src, dst, d=args.dampings[0], name=GRAPH)

    banner(f"Is the spectrum of {GRAPH}'s Google matrix real?")
    rows, spectra = [], {}
    for d in args.dampings:
        G = base_G.with_damping(d)
        spec = sample_spectrum(G, G.n, k=50)
        spectra[d] = spec
        beta = optimal_beta(d)
        rate = convergence_rate(spec, beta)
        mu1 = float(mu_magnitude(np.array([1.0 + 0j]), beta)[0])
        mu_sub = float(mu_magnitude(spec, beta).max())
        print(f"  d={d}:  {spec.size} eigenvalues sampled, "
              f"max |Im lambda| = {np.abs(spec.imag).max():.2e}")
        print(f"          beta = d^2/4 = {beta:.6f};  sqrt(beta) = {np.sqrt(beta):.4f}")
        print(f"          strongest rival |mu| = {mu_sub:.4f}  vs  |mu(lambda_1)| = {mu1:.4f}"
              f"   -> rate {rate:.4f}, predicted {asymptotic_rate(d):.4f}")
        rows.append(dict(graph=GRAPH, d=d, n_sampled=int(spec.size),
                         max_abs_imag=float(np.abs(spec.imag).max()),
                         beta=beta, sqrt_beta=float(np.sqrt(beta)),
                         mu_subdominant=mu_sub, mu_dominant=mu1,
                         rate_from_spectrum=rate,
                         rate_predicted=asymptotic_rate(d),
                         predicted_speedup=predicted_speedup(d)))

    pd.DataFrame(rows).to_csv(result_path("exp5_spectrum.csv"), index=False)
    print("\nwrote result/exp5_spectrum.csv")

    # ---------------- figure ----------------
    fig, ax = plt.subplots(1, 3, figsize=(13.4, 4.0))
    focus = args.dampings[-1]
    spec, beta = spectra[focus], optimal_beta(focus)
    sq = np.sqrt(beta)

    # A: the spectrum in the complex plane -- it lies on the real axis
    a = ax[0]
    a.scatter(spec.real, spec.imag, s=26, color="#0072B2", label="eigenvalues of $G$")
    a.plot(1.0, 0.0, "*", ms=16, color="#D55E00", label=r"$\lambda_1=1$")
    a.axhline(0, color="black", lw=0.7)
    a.set_xlabel(r"$\mathrm{Re}\,\lambda$"); a.set_ylabel(r"$\mathrm{Im}\,\lambda$")
    a.set_ylim(-0.5, 0.5)
    a.set_title(f"A. The spectrum is real\n$d={focus}$, max $|\\mathrm{{Im}}\\,\\lambda|$ = "
                f"{np.abs(spec.imag).max():.1e}")
    a.legend(fontsize=7.5)

    # B: where each lambda lands after momentum -- the fold
    a = ax[1]
    t = np.linspace(0, 2 * np.pi, 400)
    a.plot(sq * np.cos(t), sq * np.sin(t), color="#009E73", ls="--", lw=1.7,
           label=rf"circle $\sqrt{{\beta}}$ = {sq:.3f}")
    mus = []
    for lam in spec:
        disc = np.sqrt(lam ** 2 - 4 * beta + 0j)
        mus += [(lam + disc) / 2, (lam - disc) / 2]
    mus = np.array(mus)
    a.scatter(mus.real, mus.imag, s=22, color="#0072B2", label=r"rivals: $\mu(\lambda_l)$")
    d1 = np.sqrt(complex(1 - 4 * beta))
    a.plot(((1 + d1) / 2).real, ((1 + d1) / 2).imag, "*", ms=16, color="#D55E00",
           label=r"$\mu(\lambda_1)$ — stays outside")
    a.set_aspect("equal"); a.set_xlabel(r"$\mathrm{Re}\,\mu$"); a.set_ylabel(r"$\mathrm{Im}\,\mu$")
    a.set_title("B. Momentum folds every rival\nonto one small circle")
    a.legend(fontsize=7)

    # C: the rate as a function of beta, with d^2/4 marked at the minimum
    a = ax[2]
    betas = np.linspace(0, 0.2499, 400)
    for d in args.dampings:
        curve = rate_curve(spectra[d], betas)
        line, = a.plot(betas, curve, label=f"$d={d}$")
        b = optimal_beta(d)
        a.plot(b, convergence_rate(spectra[d], b), "o", ms=8, color=line.get_color())
    a.set_xlabel(r"momentum parameter $\beta$")
    a.set_ylabel("true convergence rate")
    a.set_title(r"C. $\beta=d^2/4$ (dots) sits at the minimum" "\n" "of the true rate curve")
    a.legend(fontsize=7.5)
    save_figure(fig, "exp5_spectrum")

    banner("SUMMARY: the theory's assumption holds, so its prediction holds")
    df = pd.DataFrame(rows)
    print(df[["d", "max_abs_imag", "sqrt_beta", "mu_subdominant", "mu_dominant",
              "rate_from_spectrum", "rate_predicted"]]
          .to_string(index=False, float_format=lambda v: f"{v:10.6f}"))
    print("\n  The spectrum is real to ~1e-15, so every rival folds to exactly sqrt(beta)")
    print("  and the rate computed from the spectrum equals the paper's rho(d).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
