"""Phase 6: why the paper's parameter choice does not transfer to PageRank.

Experiments 2 and 3 establish the fact -- ``beta = d^2/4`` makes the momentum
iteration diverge on real graphs -- without explaining it.  This script supplies
the mechanism, quantitatively.

The momentum iteration is a power iteration on ``A_beta``, whose eigenvalues are
``mu_+- = (lambda +- sqrt(lambda^2 - 4 beta))/2`` for each eigenvalue ``lambda``
of ``A``.  For a **real** spectrum every ``lambda`` with ``lambda^2/4 < beta``
maps to a pair of modulus exactly ``sqrt(beta)``, which is what makes
``beta = lambda_2^2/4`` optimal (Corollary 2.3).  For a **complex** ``lambda``
that identity fails upward: at ``lambda = i y`` the modulus is
``(|y| + sqrt(y^2 + 4 beta))/2 > sqrt(beta)``, growing without bound in ``|y|``.

So the destabilising modes of a non-normal matrix are not the ones near the top
of the spectrum, but small-modulus eigenvalues near the imaginary axis -- which
a largest-magnitude eigensolver never even returns.  The Google matrix has them.

Outputs
  * ``figures/exp5_rate_vs_beta``  -- true convergence rate against ``beta``,
    with ``d^2/4`` marked, showing it sits past the stability boundary.
  * ``figures/exp5_spectrum_plane`` -- the sampled spectrum in the complex
    plane, coloured by ``|mu(lambda)|``, with the modes that overtake the
    dominant one highlighted.
  * ``results/exp5_spectrum.csv``

Usage::

    python experiments/exp5_spectrum.py --graphs cit-HepPh wiki-Vote
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from _common import banner, results_path
from src.datasets import load_snap_edges, synthetic_web_graph
from src.google_matrix import build_google_matrix
from src.iterations import optimal_beta, power_iteration, predicted_speedup, static_momentum
from src.plotting import save_figure, use_style
from src.spectrum import (
    best_beta,
    imaginary_axis_limit,
    mu_magnitude,
    rate_curve,
    sample_spectrum,
    stability_limit,
)

DEFAULT_GRAPHS = ["cit-HepPh", "wiki-Vote"]
DEFAULT_DAMPINGS = [0.85, 0.95, 0.99]


def analyse(key: str, src, dst, dampings, rows: list[dict]) -> list[tuple]:
    base, _ = build_google_matrix(src, dst, d=dampings[0], name=key)
    print(f"  {base!r}")
    panels = []

    for d in dampings:
        G = base.with_damping(d)
        spec = sample_spectrum(G, G.n, k=60)
        if spec.size == 0:
            print(f"    d={d}: eigenvalue sampling failed; skipped")
            continue

        beta_paper = optimal_beta(d)
        rate_paper = rate_curve(spec, [beta_paper])[0]
        b_best, rate_best = best_beta(spec)
        b_stable = stability_limit(spec)
        b_imag = imaginary_axis_limit(d)

        # Which sampled modes overtake the dominant one at the paper's beta?
        m1 = float(mu_magnitude(np.array([1.0 + 0j]), beta_paper)[0])
        mags = mu_magnitude(spec, beta_paper)
        bad = spec[mags > m1]

        speedup_best = (np.log(rate_best) / np.log(d)) if 0 < rate_best < 1 else float("nan")

        # The rates above come from a *sample* of the spectrum, and the true
        # rate is a maximum over all of it, so a sampled rate can only be too
        # low -- the predicted acceleration is optimistic by an unknown margin.
        # Rather than leave that as a caveat, run the static method at the beta
        # the sample recommends and measure what actually happens.  Agreement
        # means the sample captured the modes that matter; disagreement means it
        # did not, and says so in the table instead of overclaiming.
        G.reset()
        emp_power = power_iteration(G, tol=1e-10, max_iter=60_000)
        G.reset()
        emp_best = static_momentum(G, beta=b_best, tol=1e-10, max_iter=60_000)
        measured = (emp_power.matvecs / emp_best.matvecs) if emp_best.converged else float("nan")
        print(f"    d={d}:")
        print(f"       paper's beta = d^2/4 = {beta_paper:.6f}  ->  true rate "
              f"{rate_paper:.4f}  {'(DIVERGES)' if rate_paper >= 1 else ''}")
        print(f"       largest stable beta   = {b_stable:.6f}")
        print(f"       best beta             = {b_best:.6f}  ->  rate {rate_best:.4f} "
              f"vs power {d:.4f}   ({speedup_best:.2f}x achievable, "
              f"{predicted_speedup(d):.2f}x predicted by real-spectrum theory)")
        print(f"       worst-case-safe beta  = {b_imag:.6f}  (bound over |lambda| <= d)")
        print(f"       sampled modes overtaking the dominant one: {bad.size} of {spec.size}")
        print(f"       EMPIRICAL CHECK at beta_best: power {emp_power.matvecs} mv vs "
              f"static {emp_best.matvecs} mv"
              f"{'' if emp_best.converged else ' (did not converge)'}"
              f"  ->  {measured:.2f}x measured vs {speedup_best:.2f}x predicted from the sample"
              f"   [{'sample adequate' if emp_best.converged and measured > 0.7 * speedup_best else 'SAMPLE INCOMPLETE: prediction optimistic'}]")
        if bad.size:
            worst = bad[np.argmax(mu_magnitude(bad, beta_paper))]
            print(f"       worst: lambda = {worst.real:+.4f}{worst.imag:+.4f}j  "
                  f"|lambda| = {abs(worst):.4f}, arg = {np.angle(worst) / np.pi:+.3f}pi, "
                  f"|mu| = {mu_magnitude(np.array([worst]), beta_paper)[0]:.4f} > {m1:.4f}")

        rows.append(dict(graph=key, n=G.n, d=d, n_sampled=int(spec.size),
                         max_abs_imag=float(np.abs(spec.imag).max()),
                         beta_paper=beta_paper, rate_at_beta_paper=float(rate_paper),
                         diverges=bool(rate_paper >= 1.0),
                         beta_stability_limit=b_stable, beta_best=b_best,
                         rate_best=rate_best, achievable_speedup=speedup_best,
                         predicted_speedup=predicted_speedup(d),
                         beta_imaginary_axis_bound=b_imag,
                         n_modes_overtaking=int(bad.size),
                         measured_speedup_at_beta_best=measured,
                         matvecs_power=emp_power.matvecs,
                         matvecs_at_beta_best=emp_best.matvecs,
                         converged_at_beta_best=bool(emp_best.converged)))
        panels.append((d, spec, beta_paper, b_best, b_stable))
    return panels


def plot_rate_vs_beta(all_panels, dampings) -> None:
    import matplotlib.pyplot as plt

    n = len(all_panels)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 3.7), squeeze=False)
    for ax, (key, panels) in zip(axes[0], all_panels):
        for d, spec, beta_paper, b_best, b_stable in panels:
            betas = np.linspace(0, 0.2499, 500)
            ax.plot(betas, rate_curve(spec, betas), label=f"$d={d}$")
            ax.plot(beta_paper, rate_curve(spec, [beta_paper])[0], "x",
                    color=ax.lines[-1].get_color(), ms=8, mew=2)
        ax.axhline(1.0, color="red", ls=":", lw=1.2)
        ax.text(0.005, 1.02, "divergence", color="red", fontsize=7)
        ax.set_xlabel(r"momentum parameter $\beta$")
        ax.set_ylabel("true convergence rate")
        ax.set_title(f"{key}\n(x marks the paper's $\\beta=d^2/4$)")
        ax.set_ylim(0.6, 1.3)
        ax.legend(fontsize=8)
    save_figure(fig, "exp5_rate_vs_beta")


def plot_spectrum_plane(all_panels) -> None:
    import matplotlib.pyplot as plt

    items = [(k, p) for k, panels in all_panels for p in panels if p[0] == 0.99]
    if not items:
        items = [(k, panels[-1]) for k, panels in all_panels]
    fig, axes = plt.subplots(1, len(items), figsize=(4.6 * len(items), 4.0),
                             squeeze=False)
    for ax, (key, (d, spec, beta_paper, _, _)) in zip(axes[0], items):
        mags = mu_magnitude(spec, beta_paper)
        m1 = float(mu_magnitude(np.array([1.0 + 0j]), beta_paper)[0])
        sc = ax.scatter(spec.real, spec.imag, c=mags, cmap="viridis", s=18)
        bad = spec[mags > m1]
        if bad.size:
            ax.scatter(bad.real, bad.imag, facecolors="none", edgecolors="red",
                       s=70, lw=1.2, label=r"$|\mu|>|\mu_{\lambda_1}|$: diverging")
        circ = np.exp(1j * np.linspace(0, 2 * np.pi, 400))
        ax.plot((d * circ).real, (d * circ).imag, "k--", lw=0.9,
                label=r"$|\lambda|=d$")
        ax.set_aspect("equal")
        ax.set_xlabel(r"$\mathrm{Re}\,\lambda$")
        ax.set_ylabel(r"$\mathrm{Im}\,\lambda$")
        ax.set_title(f"{key}, $d={d}$, $\\beta=d^2/4$")
        ax.legend(fontsize=7, loc="upper right")
        fig.colorbar(sc, ax=ax, label=r"$|\mu(\lambda)|$")
    save_figure(fig, "exp5_spectrum_plane")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", nargs="*", default=DEFAULT_GRAPHS)
    parser.add_argument("--dampings", nargs="*", type=float, default=DEFAULT_DAMPINGS)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    use_style()
    rows: list[dict] = []
    all_panels = []

    banner("Worst-case-safe beta over the disk |lambda| <= d (no graph needed)")
    print(f"  {'d':>7}  {'beta = d^2/4':>13}  {'safe beta':>11}  ratio")
    for d in args.dampings:
        b = imaginary_axis_limit(d)
        print(f"  {d:>7.3f}  {optimal_beta(d):>13.6f}  {b:>11.6f}  "
              f"{optimal_beta(d) / b:>6.1f}x too large")

    for key in args.graphs:
        banner(f"Spectral diagnosis: {key}")
        try:
            if args.offline:
                raise RuntimeError("offline mode")
            src, dst = load_snap_edges(key)
        except Exception as exc:                    # noqa: BLE001
            print(f"  {key} unavailable ({exc}); using a synthetic graph")
            src, dst = synthetic_web_graph(n=20_000, seed=abs(hash(key)) % 2**31)
        panels = analyse(key, src, dst, args.dampings, rows)
        if panels:
            all_panels.append((key, panels))

    if all_panels:
        plot_rate_vs_beta(all_panels, args.dampings)
        plot_spectrum_plane(all_panels)

    df = pd.DataFrame(rows)
    df.to_csv(results_path("exp5_spectrum.csv"), index=False)
    print("\nwrote results/exp5_spectrum.csv")

    banner("SUMMARY: achievable vs predicted acceleration")
    cols = ["graph", "d", "rate_at_beta_paper", "diverges", "beta_best",
            "achievable_speedup", "measured_speedup_at_beta_best",
            "predicted_speedup"]
    print(df[cols].to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
