"""Phase 1 gate: reproduce the base paper before touching PageRank.

If Algorithm 3.1 does not behave on the paper's own benchmarks the way the
paper says it does, every later result is meaningless.  This script therefore
runs first and prints an explicit PASS/FAIL verdict.

It produces
  * ``figures/fig1_rate_comparison`` -- Figure 1 of the paper: the accelerated
    rate ``rho(r) = r/(1+sqrt(1-r^2))`` against ``r^p``.
  * ``figures/fig3_matrix1`` -- Figure 3 (left): convergence on
    ``A = diag(1000:-1:1)``.
  * ``figures/exp1_suitesparse`` -- the same comparison on the SuiteSparse
    matrices of test suites 1 and 2, when they can be downloaded.
  * ``results/exp1_reproduce.csv`` -- matrix-vector counts per method.

Usage::

    python experiments/exp1_reproduce.py            # full
    python experiments/exp1_reproduce.py --quick    # skip SuiteSparse + random starts
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from _common import banner, results_path
from src.datasets import PAPER_MATRICES, diag_benchmark, load_suitesparse
from src.google_matrix import CountingOperator
from src.iterations import (
    asymptotic_rate,
    dynamic_momentum,
    optimal_beta,
    power_iteration,
    predicted_speedup,
    static_momentum,
)
from src.plotting import METHOD_STYLE, markevery, save_figure, use_style

TOL = 1e-12
MAX_ITER = 2000


# --------------------------------------------------------------------------
# Figure 1: the accelerated rate against powers of r
# --------------------------------------------------------------------------


def figure_rate_comparison() -> None:
    import matplotlib.pyplot as plt

    r = np.linspace(1e-3, 0.9999, 2000)
    rho = np.array([asymptotic_rate(v) for v in r])

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for ax, powers, lo in ((axes[0], (1, 2, 3, 4, 6, 10), 0.0),
                           (axes[1], (6, 10, 14, 20), 0.9)):
        mask = r >= lo
        ax.plot(r[mask], rho[mask], color="black", lw=2,
                label=r"$\rho(r)=r/(1+\sqrt{1-r^2})$")
        for p in powers:
            ax.plot(r[mask], r[mask] ** p, lw=1.0, alpha=0.8, label=f"$r^{{{p}}}$")
            # Mark where rho(r) crosses r^p: below this r the momentum method
            # beats a p-fold reduction of the power iteration's rate.
            diff = rho[mask] - r[mask] ** p
            sign = np.sign(diff)
            cross = np.flatnonzero(np.diff(sign) != 0)
            if cross.size:
                i = cross[-1]
                ax.plot(r[mask][i], rho[mask][i], "k.", ms=6)
        ax.set_xlabel("$r = |\\lambda_2/\\lambda_1|$")
        ax.set_xlim(lo, 1.0)
        ax.legend(ncol=2)
    axes[0].set_ylabel("convergence rate")
    axes[0].set_title("Accelerated rate vs. $r^p$")
    axes[1].set_title("Detail near $r \\to 1$")
    axes[1].set_ylim(0, 0.45)
    fig.suptitle("Reproduction of Figure 1 (Austin, Pollock & Zhu 2024)", y=1.02)
    save_figure(fig, "fig1_rate_comparison")


# --------------------------------------------------------------------------
# Runner shared by every benchmark matrix
# --------------------------------------------------------------------------


def run_all_methods(A, lambda2: float, v0=None, tol: float = TOL,
                    max_iter: int = MAX_ITER) -> dict:
    """Run the three methods on one matrix and return their results."""
    beta = optimal_beta(lambda2)
    out = {}

    op = CountingOperator(A)
    out["power"] = power_iteration(op, v0=v0, tol=tol, max_iter=max_iter)
    op = CountingOperator(A)
    out["static"] = static_momentum(op, beta=beta, v0=v0, tol=tol, max_iter=max_iter)
    op = CountingOperator(A)
    out["dynamic"] = dynamic_momentum(op, v0=v0, tol=tol, max_iter=max_iter)
    return out


def leading_eigenvalues(A, k: int = 4, symmetric: bool = True) -> np.ndarray:
    """Largest-magnitude eigenvalues, sorted by magnitude descending.

    ARPACK starts from a random vector unless one is supplied, so repeated runs
    return ``lambda_2`` to slightly different accuracy.  That feeds straight into
    ``beta = lambda_2^2 / 4`` and moves the static method's iteration count by a
    percent or two between runs.  Fixing the start vector makes the reported
    numbers reproducible.
    """
    from scipy.sparse.linalg import eigs, eigsh

    k = min(k, A.shape[0] - 2)
    v0 = np.ones(A.shape[0])
    if symmetric:
        vals = eigsh(A, k=k, which="LM", v0=v0, return_eigenvectors=False)
    else:
        vals = eigs(A, k=k, which="LM", v0=v0, return_eigenvectors=False)
    vals = np.asarray(vals)
    return vals[np.argsort(-np.abs(vals))]


def effective_lambda2(vals: np.ndarray, rtol: float = 1e-10) -> float:
    """First eigenvalue strictly smaller in magnitude than ``lambda_1``.

    Matrix 3 (``Muu``) of the paper has ``lambda_2 = lambda_1`` to machine
    precision, and the static method fails outright with ``beta = lambda_2^2/4``
    because the augmented matrix then has no spectral gap.  The paper works
    around this by hand, substituting ``lambda_3``; doing it automatically keeps
    the driver uniform across matrices.
    """
    top = abs(vals[0])
    for v in vals[1:]:
        if abs(abs(v) - top) > rtol * top:
            return float(abs(v))
    return float(abs(vals[-1]))


def plot_convergence(ax, results: dict, title: str, x: str = "iteration") -> None:
    for key, res in results.items():
        style = METHOD_STYLE.get(key, {})
        y = np.asarray(res.residual)
        xs = np.arange(1, y.size + 1)
        ax.semilogy(xs, y, markevery=markevery(y.size), **style)
    ax.set_xlabel(x)
    ax.set_ylabel(r"residual $\|Ax_k-\nu_k x_k\|_2$")
    ax.set_title(title)


# --------------------------------------------------------------------------
# Matrix 1: diag(1000:-1:1)
# --------------------------------------------------------------------------


def matrix1_experiment(rows: list[dict]) -> dict:
    import matplotlib.pyplot as plt

    banner("Matrix 1:  A = diag(1000 : -1 : 1)   (test suite 1, r = 0.999)")
    n = 1000
    A, eigenvalues = diag_benchmark(n)
    lambda1, lambda2 = eigenvalues[0], eigenvalues[1]
    r = lambda2 / lambda1
    print(f"  lambda_1 = {lambda1:.6g}, lambda_2 = {lambda2:.6g}, r = {r:.6f}")
    print(f"  theory: accelerated rate rho = {asymptotic_rate(r):.6f}, "
          f"predicted speedup = {predicted_speedup(r):.2f}x")

    results = run_all_methods(A, lambda2)
    for key, res in results.items():
        print(f"  {key:>8s}: {res.matvecs:5d} matvecs, "
              f"final residual {res.final_residual():.3e}, "
              f"converged={res.converged}")
        rows.append(dict(matrix="diag(1000:-1:1)", n=n, r=r, method=key,
                         matvecs=res.matvecs, iterations=res.n_iter,
                         converged=res.converged, residual=res.final_residual(),
                         time_s=res.time, safeguard_hits=res.safeguard_hits))

    # The paper caps every run at 2000 iterations, and Figure 3 accordingly
    # shows the power iteration still far from the tolerance when it stops.
    # That cap makes the *measured* speedup meaningless, so run the power
    # iteration once more without it to obtain its true cost.
    uncapped = power_iteration(A, tol=TOL, max_iter=100_000)
    print(f"  power (uncapped): {uncapped.matvecs} matvecs, converged={uncapped.converged}")
    print(f"  measured speedup vs power:  static {uncapped.matvecs / results['static'].matvecs:.2f}x, "
          f"dynamic {uncapped.matvecs / results['dynamic'].matvecs:.2f}x  "
          f"(theory {predicted_speedup(r):.2f}x)")
    rows.append(dict(matrix="diag(1000:-1:1)", n=n, r=r, method="power_uncapped",
                     matvecs=uncapped.matvecs, iterations=uncapped.n_iter,
                     converged=uncapped.converged, residual=uncapped.final_residual(),
                     time_s=uncapped.time, safeguard_hits=0))
    results["_power_uncapped"] = uncapped

    plotted = {k: v for k, v in results.items() if not k.startswith("_")}
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    plot_convergence(axes[0], plotted, "Matrix 1: diag(1000:-1:1)")
    axes[0].legend()

    # The beta_k trace: does the dynamic method find beta_opt on its own?
    dyn = results["dynamic"]
    beta_opt = optimal_beta(lambda2)
    axes[1].plot(np.arange(1, len(dyn.beta) + 1), dyn.beta,
                 color=METHOD_STYLE["dynamic"]["color"], label=r"$\beta_k$ (Alg. 3.1)")
    axes[1].axhline(beta_opt, color="black", ls="--", lw=1.2,
                    label=r"$\beta_{opt}=\lambda_2^2/4$")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel(r"$\beta_k$")
    axes[1].set_title(r"Dynamic $\beta_k$ discovers $\beta_{opt}$ without spectral data")
    axes[1].legend()
    save_figure(fig, "fig3_matrix1")
    return results


# --------------------------------------------------------------------------
# SuiteSparse benchmarks
# --------------------------------------------------------------------------


def suitesparse_experiment(rows: list[dict]) -> None:
    import matplotlib.pyplot as plt

    banner("SuiteSparse benchmarks from the paper's test suites 1 and 2")
    panels = []
    for name, meta in PAPER_MATRICES.items():
        try:
            A = load_suitesparse(name)
        except Exception as exc:                      # noqa: BLE001
            print(f"  {name}: unavailable ({type(exc).__name__}: {exc}) -- skipped")
            continue

        vals = leading_eigenvalues(A, k=5, symmetric=True)
        lambda1 = abs(vals[0])
        lambda2 = effective_lambda2(vals)
        r = lambda2 / lambda1
        print(f"  {name}: n={A.shape[0]}, lambda_1={lambda1:.6g}, "
              f"lambda_2={lambda2:.6g}, r={r:.4f} (paper reports r={meta['r']})")

        results = run_all_methods(A, lambda2)
        for key, res in results.items():
            print(f"    {key:>8s}: {res.matvecs:5d} matvecs, "
                  f"residual {res.final_residual():.3e}, converged={res.converged}")
            rows.append(dict(matrix=name, n=A.shape[0], r=r, method=key,
                             matvecs=res.matvecs, iterations=res.n_iter,
                             converged=res.converged, residual=res.final_residual(),
                             time_s=res.time, safeguard_hits=res.safeguard_hits))
        panels.append((name, r, results))

    if not panels:
        print("  No SuiteSparse matrices available; skipping figure.")
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.6),
                             squeeze=False)
    for ax, (name, r, results) in zip(axes[0], panels):
        plot_convergence(ax, results, f"{name}  ($r={r:.4f}$)")
    axes[0][0].legend()
    save_figure(fig, "exp1_suitesparse")


def random_start_experiment(rows: list[dict], n_runs: int = 100) -> None:
    """Table 1 of the paper: sensitivity of each method to the initial vector."""
    banner(f"Sensitivity to the initial vector ({n_runs} random starts)")
    rng = np.random.default_rng(42)

    targets = []
    A, vals = diag_benchmark(200)
    targets.append(("diag(200:-1:1)", A, float(vals[1] / vals[0]), float(vals[1])))

    import scipy.sparse as sp
    v = np.linspace(-99, 100, 200)
    v = v[np.argsort(-np.abs(v))]
    targets.append(("diag(linspace(-99,100,200))", sp.diags(np.linspace(-99, 100, 200)).tocsr(),
                    99 / 100, 99.0))

    for name, A, r, lambda2 in targets:
        stats: dict[str, list[int]] = {"power": [], "static": [], "dynamic": []}
        for _ in range(n_runs):
            v0 = rng.random(A.shape[0]) - 0.5
            res = run_all_methods(A, lambda2, v0=v0)
            for key, out in res.items():
                stats[key].append(out.matvecs)
        print(f"  {name}  (r = {r:.4f})")
        for key, counts in stats.items():
            arr = np.asarray(counts)
            print(f"    {key:>8s}: min {arr.min():5d}  max {arr.max():5d}  "
                  f"mean {arr.mean():8.2f}  std {arr.std():7.2f}")
            rows.append(dict(matrix=name, n=A.shape[0], r=r, method=key,
                             matvecs=float(arr.mean()), matvecs_min=int(arr.min()),
                             matvecs_max=int(arr.max()), matvecs_std=float(arr.std()),
                             n_runs=n_runs, experiment="random_start"))


# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="skip SuiteSparse downloads and the random-start study")
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()

    use_style()
    rows: list[dict] = []

    figure_rate_comparison()
    results = matrix1_experiment(rows)

    if not args.quick:
        suitesparse_experiment(rows)
        random_start_experiment(rows, n_runs=args.runs)

    df = pd.DataFrame(rows)
    path = results_path("exp1_reproduce.csv")
    df.to_csv(path, index=False)
    print(f"\nwrote {path.relative_to(path.parent.parent)}")

    # ---------------- gate ----------------
    banner("PHASE 1 GATE")
    static, dynamic = results["static"], results["dynamic"]
    power = results["_power_uncapped"]
    measured = power.matvecs / dynamic.matvecs
    theory = predicted_speedup(0.999)
    checks = [
        ("dynamic momentum converges on Matrix 1", dynamic.converged),
        (f"dynamic beats plain power by >5x ({measured:.1f}x measured)",
         measured > 5.0),
        (f"measured speedup is within 25% of the predicted {theory:.1f}x",
         abs(measured - theory) < 0.25 * theory),
        ("dynamic is at least as fast as the optimal static method",
         dynamic.matvecs <= static.matvecs),
        ("dynamic beta_k settles near beta_opt",
         abs(np.median(dynamic.beta[-50:]) - optimal_beta(999.0))
         < 0.05 * optimal_beta(999.0)),
    ]
    ok = True
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        ok &= bool(passed)
    print(f"\n  Phase 1 gate: {'PASS -- proceed to Phase 2' if ok else 'FAIL -- fix the solver first'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
