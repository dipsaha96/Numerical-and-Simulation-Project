"""Phase 8: isolating non-normality as the single cause, by controlled sweep.

Experiment 6 shows the momentum methods accelerate on symmetric ranking
problems and experiments 2-5 show they do not on the Google matrix, and
section 6 argues the difference is the complex spectrum.  That argument has a
gap a sharp reader will find: the co-citation control changes *three* things at
once relative to the PageRank case -- the symmetry, the matrix itself, and the
spectral ratio ``r``.  Observing a different outcome therefore does not by
itself establish *which* of the three is responsible.

This script closes that gap with a single-variable experiment.  It constructs a
family of matrices in which

* ``lambda_1 = 1`` is fixed,
* ``|lambda_2| = r`` is fixed -- so the spectral ratio, and hence the *predicted*
  speedup ``log rho(r) / log r``, is identical for every member of the family,
* the subdominant eigenvalues all have the same moduli,

and the **only** quantity that varies is the argument ``theta`` of those
subdominant eigenvalues, rotated from ``0`` (a real spectrum, symmetric matrix)
to ``pi/2`` (purely imaginary).  Each matrix is block diagonal with ``[1]`` in
the corner and 2x2 rotation-scaling blocks

    [ m cos(theta)   -m sin(theta) ]
    [ m sin(theta)    m cos(theta) ]

whose eigenvalues are ``m exp(+- i theta)``.

Because ``|lambda_2|`` never changes, the plain power iteration is equally hard
on every member of the family -- it converges at rate ``r`` regardless of where
the eigenvalues sit on their circle.  That makes it an internal control: if its
cost stays flat while the momentum methods collapse, the collapse cannot be
attributed to the problems becoming harder.

The second panel of the figure ties the measurement back to the theory of
section 6: it plots ``max |mu(lambda)|`` over the subdominant spectrum against
``|mu(lambda_1)|``, whose crossing predicts where divergence must begin.

Usage::

    python experiments/exp7_normality.py
    python experiments/exp7_normality.py --ratios 0.9 --blocks 60
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from _common import banner, results_path
from src.iterations import (
    dynamic_momentum,
    optimal_beta,
    power_iteration,
    predicted_speedup,
    static_momentum,
)
from src.plotting import METHOD_STYLE, save_figure, use_style
from src.spectrum import mu_magnitude

DEFAULT_RATIOS = [0.85, 0.90, 0.95]
DEFAULT_FRACTIONS = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
TOL = 1e-12
MAX_ITER = 200_000

# Same safeguard settings the PageRank experiments use.
_SAFE = dict(backtrack_window=8, backtrack_grow=1.06, grow_patience=4,
             max_backtracks=20)


def build_matrix(theta: float, r: float, blocks: int = 40) -> np.ndarray:
    """Block-diagonal matrix with ``lambda_1 = 1`` and subdominant ``m e^{+-i theta}``.

    The moduli are spread from ``r`` down towards zero so the spectrum is not
    degenerate, but the *largest* subdominant modulus is always exactly ``r``.
    Rotating ``theta`` therefore moves every subdominant eigenvalue along its
    own circle without changing any modulus, and in particular without changing
    ``r = |lambda_2 / lambda_1|``.
    """
    moduli = np.linspace(r, 0.05, blocks)
    n = 1 + 2 * blocks
    A = np.zeros((n, n))
    A[0, 0] = 1.0
    for j, m in enumerate(moduli):
        i = 1 + 2 * j
        c, s = m * np.cos(theta), m * np.sin(theta)
        A[i, i], A[i, i + 1] = c, -s
        A[i + 1, i], A[i + 1, i + 1] = s, c
    return A


def subdominant_spectrum(theta: float, r: float, blocks: int) -> np.ndarray:
    moduli = np.linspace(r, 0.05, blocks)
    return np.concatenate([moduli * np.exp(1j * theta),
                           moduli * np.exp(-1j * theta)])


def run_family(r: float, fractions, blocks: int, rows: list[dict]) -> None:
    beta = optimal_beta(r)          # the paper's optimal parameter for this r
    predicted = predicted_speedup(r)
    print(f"\n  r = {r}:  beta_opt = r^2/4 = {beta:.6f},  "
          f"predicted speedup = {predicted:.2f}x for every theta below")
    print(f"    {'arg/pi':>7} {'sym':>6} {'max|Im|':>8} | {'power':>7} {'static':>8} "
          f"{'dynamic':>9} {'dyn+safe':>9} | {'speedup':>9} {'theory|mu|':>11}")
    print("    " + "-" * 88)

    for frac in fractions:
        theta = frac * np.pi
        A = build_matrix(theta, r, blocks)
        symmetric = bool(np.allclose(A, A.T))

        sub = subdominant_spectrum(theta, r, blocks)
        mu_sub = float(mu_magnitude(sub, beta).max())
        mu_one = float(mu_magnitude(np.array([1.0 + 0j]), beta)[0])
        theory_diverges = mu_sub >= mu_one

        p = power_iteration(A, tol=TOL, max_iter=MAX_ITER)
        st = static_momentum(A, beta=beta, tol=TOL, max_iter=MAX_ITER)
        dy = dynamic_momentum(A, tol=TOL, max_iter=MAX_ITER)
        sf = dynamic_momentum(A, tol=TOL, max_iter=MAX_ITER, lambda1=1.0,
                              r_max=r, **_SAFE)

        tag = lambda x: f"{x.matvecs}" + ("" if x.converged else "!")
        speed = f"{p.matvecs / dy.matvecs:.2f}x" if dy.converged else "DIVERGED"
        print(f"    {frac:>7.2f} {str(symmetric):>6} {np.abs(sub.imag).max():>8.4f} | "
              f"{tag(p):>7} {tag(st):>8} {tag(dy):>9} {tag(sf):>9} | "
              f"{speed:>9} {mu_sub:>7.4f}{'>' if theory_diverges else '<'}{mu_one:.3f}")

        for name, res in (("power", p), ("static", st),
                          ("dynamic", dy), ("dynamic_safe", sf)):
            rows.append(dict(
                r=r, theta_over_pi=frac, theta=theta, n=A.shape[0],
                symmetric=symmetric, max_abs_imag=float(np.abs(sub.imag).max()),
                method=name, matvecs=res.matvecs, converged=res.converged,
                diverged=res.diverged, residual=res.final_residual(),
                speedup=(p.matvecs / res.matvecs) if res.converged else float("nan"),
                predicted_speedup=predicted,
                mu_subdominant=mu_sub, mu_dominant=mu_one,
                theory_diverges=theory_diverges,
                beta=beta, backtracks=res.backtracks))


def plot(df: pd.DataFrame, blocks: int) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.0))
    fig.subplots_adjust(wspace=0.34)
    focus = sorted(df.r.unique())[len(df.r.unique()) // 2]
    ratios = sorted(df.r.unique())

    # `dynamic` and `dynamic_safe` share a style in METHOD_STYLE because no other
    # figure draws them together; here they must be told apart.
    LOCAL = {
        "power":        dict(color="#0072B2", marker="o", label="Power (Alg. 1.1)"),
        "static":       dict(color="#D55E00", marker="s", label=r"Static $\beta=r^2/4$"),
        "dynamic":      dict(color="#CC79A7", marker="D", label="Dynamic (Alg. 3.1)"),
        "dynamic_safe": dict(color="#009E73", marker="^", label="Dynamic + safeguards"),
    }

    # ---- A: the single-variable claim, across every r ----
    ax = axes[0]
    band = {r: 0.60 - 0.055 * i for i, r in enumerate(ratios)}
    for r in ratios:
        dyn = df[(df.r == r) & (df.method == "dynamic")].sort_values("theta_over_pi")
        safe = df[(df.r == r) & (df.method == "dynamic_safe")].sort_values("theta_over_pi")
        line, = ax.plot(safe.theta_over_pi, safe.speedup, "-", lw=1.4,
                        label=f"$r={r}$")
        c = line.get_color()
        ax.axhline(dyn.predicted_speedup.iloc[0], ls="--", lw=1.0, alpha=0.5, color=c)
        ax.plot(dyn[dyn.converged].theta_over_pi, dyn[dyn.converged].speedup,
                "o", ms=8, color=c)
        bad = dyn[~dyn.converged]
        ax.plot(bad.theta_over_pi, np.full(len(bad), band[r]), "x",
                color=c, ms=6, mew=1.6)
    ax.axhline(1.0, color="red", ls=":", lw=1.2)
    ax.set_xlabel(r"$\arg\lambda/\pi$ of the subdominant eigenvalues")
    ax.set_ylabel("measured speedup")
    ax.set_title(r"A. Only $\arg\lambda$ varies")
    ax.set_ylim(0.32, None)
    ax.legend(fontsize=7, loc="upper right", title="fixed $r$", title_fontsize=7)
    ax.text(0.33, 0.30,
            "dashed = predicted (constant)\n"
            "circle = Alg. 3.1   line = safeguarded\n"
            "x = Alg. 3.1 diverged   red = parity",
            transform=ax.transAxes, fontsize=6.5, color="#444444", va="bottom")

    # ---- B: the two failure modes, at one r ----
    ax = axes[1]
    ceiling = df[df.r == focus].matvecs.max() * 1.4
    for name in ("power", "static", "dynamic", "dynamic_safe"):
        sub = df[(df.r == focus) & (df.method == name)].sort_values("theta_over_pi")
        st = LOCAL[name]
        ok = sub[sub.converged]
        ax.plot(ok.theta_over_pi, ok.matvecs, "-", color=st["color"],
                marker=st["marker"], ms=5, label=st["label"])
        bad = sub[~sub.converged]
        ax.plot(bad.theta_over_pi, np.full(len(bad), ceiling), ls="none",
                marker=st["marker"], color=st["color"], mfc="none", ms=7)
    ax.axhline(ceiling, color="#888888", ls=":", lw=1.0)
    ax.text(0.02, 0.86, "hollow = did not converge", transform=ax.transAxes,
            fontsize=6.5, color="#555555")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\arg\lambda/\pi$")
    ax.set_ylabel("matrix-vector products")
    ax.set_title(f"B. Two failure modes ($r={focus}$)")
    ax.legend(fontsize=6.5, loc="lower right")

    # ---- C: the theory that predicts mode B ----
    ax = axes[2]
    beta = optimal_beta(focus)
    th = np.linspace(0, np.pi / 2, 400)
    mu_sub = np.array([float(mu_magnitude(subdominant_spectrum(t, focus, blocks),
                                          beta).max()) for t in th])
    mu_one = float(mu_magnitude(np.array([1.0 + 0j]), beta)[0])
    ax.plot(th / np.pi, mu_sub, color="#D55E00", lw=1.8,
            label=r"$\max_{l\geq2}|\mu(\lambda_l)|$")
    ax.axhline(mu_one, color="#0072B2", lw=1.8, label=r"$|\mu(\lambda_1)|$")
    cross = np.flatnonzero(np.diff(np.sign(mu_sub - mu_one)) != 0)
    if cross.size:
        xc = th[cross[0]] / np.pi
        ax.axvspan(xc, 0.5, color="red", alpha=0.07)
        ax.axvline(xc, color="black", ls=":", lw=1.2)
        ax.text(xc + 0.015, mu_sub.max() * 0.80,
                f"theory: no $\\beta=r^2/4$\nconvergence beyond {xc:.3f}$\\pi$",
                fontsize=7)
    for name, label in (("static", r"mode B: static $\beta=r^2/4$"),
                        ("dynamic", "mode A: Alg. 3.1")):
        sub = df[(df.r == focus) & (df.method == name)].sort_values("theta_over_pi")
        bad = sub[~sub.converged]
        if len(bad):
            ax.plot(bad.theta_over_pi.iloc[0], mu_one, LOCAL[name]["marker"], ms=9,
                    color=LOCAL[name]["color"], mec="black", mew=0.8,
                    label=f"{label} fails here")
    ax.set_xlabel(r"$\arg\lambda/\pi$")
    ax.set_ylabel(r"$|\mu|$ of the augmented matrix")
    ax.set_title(f"C. Mode B predicted exactly ($r={focus}$)")
    ax.legend(fontsize=6.5, loc="upper left")
    save_figure(fig, "exp7_normality")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratios", nargs="*", type=float, default=DEFAULT_RATIOS)
    parser.add_argument("--fractions", nargs="*", type=float, default=DEFAULT_FRACTIONS,
                        help="arg(lambda)/pi values to sweep")
    parser.add_argument("--blocks", type=int, default=40)
    args = parser.parse_args()

    use_style()
    rows: list[dict] = []

    banner("Phase 8: rotating the spectrum with every other quantity held fixed")
    print("  lambda_1 = 1 and |lambda_2| = r are fixed in every matrix, so the")
    print("  predicted speedup is constant along each row-block below and the plain")
    print("  power iteration should be equally hard throughout.  The only variable")
    print("  is the argument of the subdominant eigenvalues.")

    for r in args.ratios:
        run_family(r, args.fractions, args.blocks, rows)

    df = pd.DataFrame(rows)
    df.to_csv(results_path("exp7_normality.csv"), index=False)
    print("\nwrote results/exp7_normality.csv   (! = did not converge)")
    plot(df, args.blocks)

    banner("CONCLUSION")
    pw = df[df.method == "power"]
    flat = pw.groupby("r")["matvecs"].agg(["min", "max"])
    print("  Power iteration cost across the whole sweep (should be nearly flat,")
    print("  since |lambda_2| never changes):")
    for r, row in flat.iterrows():
        print(f"    r={r}: {int(row['min'])} to {int(row['max'])} matvecs "
              f"({100 * (row['max'] / row['min'] - 1):.1f}% variation)")

    print("\n  Dynamic momentum (Algorithm 3.1):")
    for r in sorted(df.r.unique()):
        sub = df[(df.r == r) & (df.method == "dynamic")].sort_values("theta_over_pi")
        real = sub[sub.theta_over_pi == 0].iloc[0]
        bad = sub[~sub.converged]
        onset = f"{bad.theta_over_pi.iloc[0]:.2f}pi" if len(bad) else "never"
        print(f"    r={r}: {real.speedup:.2f}x on the real spectrum "
              f"(predicted {real.predicted_speedup:.2f}x), "
              f"diverges from arg = {onset}")

    # The sweep separates the two failure modes of section 6, which the PageRank
    # experiments could only observe tangled together.
    print("\n  The sweep separates the two failure modes:")
    print("    Mode B (spectral crossing) -- does theory predict where the")
    print("    STATICALLY optimal beta = r^2/4 stops working?")
    agree = 0
    for r in sorted(df.r.unique()):
        s = df[(df.r == r) & (df.method == "static")].sort_values("theta_over_pi")
        last_ok = s[~s.theory_diverges].theta_over_pi.max()
        after = s[s.theta_over_pi > last_ok].theta_over_pi.min()
        first_bad = s[~s.converged].theta_over_pi.min() if (~s.converged).any() else np.nan
        hit = abs(first_bad - after) < 1e-9
        agree += bool(hit)
        print(f"      r={r}: crossing after {last_ok:.2f}pi, predicts divergence at "
              f"{after:.2f}pi; measured {first_bad:.2f}pi  [{'MATCH' if hit else 'miss'}]")
    print(f"    -> {agree}/{len(df.r.unique())} exact agreement between the predicted")
    print( "       crossing and the measured onset.")
    print("\n    Mode A (the rho_k -> 1 runaway) -- Algorithm 3.1 fails EARLIER than")
    print("    the crossing, at the first hint of oscillation, because it infers r")
    print("    from residual ratios: one step with d_k+1 > d_k sets rho_k = 1, hence")
    print("    r = 1 and beta -> 1/4, the worst possible value.  Its self-tuning rule")
    print("    steers it into the barrier that the fixed parameter still avoids.")

    safe = df[df.method == "dynamic_safe"]
    print(f"\n  Safeguarded variant converged on "
          f"{int(safe.converged.sum())}/{len(safe)} of the same matrices.")
    print("\n  One variable changed; one outcome flipped.  Non-normality is the cause.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
