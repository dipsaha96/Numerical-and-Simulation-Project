"""Experiment 1 - reproduce the base paper, before trusting anything else.

Every later result rests on our implementation being faithful, so this runs the
three published algorithms on the paper's own benchmark matrices and checks the
numbers against what the paper reports.  It ends with an explicit PASS/FAIL and
exits non-zero on failure.

    python experiment/exp1_reproduce.py
    python experiment/exp1_reproduce.py --quick    # skip the SuiteSparse downloads
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from _common import MAX_ITER, TOL, banner, result_path, save_figure
from _methods import METHOD_LABEL, run_all_methods
from _style import STYLE, markevery, use_style
from src.datasets import PAPER_MATRICES, diag_benchmark, load_suitesparse
from src.iterations import asymptotic_rate, optimal_beta, power_iteration, predicted_speedup

# What the paper reports, for comparison.
PAPER_R = {"Kuu": 0.9981, "Muu": 0.9992, "ash292": 0.9153, "bcspwr06": 0.9814}


def leading_eigenvalues(A, k: int = 5) -> np.ndarray:
    """Largest-magnitude eigenvalues, from a fixed start so runs are repeatable."""
    from scipy.sparse.linalg import eigsh
    vals = eigsh(A, k=min(k, A.shape[0] - 2), which="LM",
                 v0=np.ones(A.shape[0]), return_eigenvectors=False)
    return np.asarray(vals)[np.argsort(-np.abs(np.asarray(vals)))]


def effective_lambda2(vals: np.ndarray, rtol: float = 1e-10) -> float:
    """First eigenvalue strictly below |lambda_1|.

    Matrix Muu has lambda_2 = lambda_1 to machine precision; the paper handles
    this by substituting lambda_3 for the static method.  Detecting it keeps the
    driver uniform across matrices.
    """
    top = abs(vals[0])
    for v in vals[1:]:
        if abs(abs(v) - top) > rtol * top:
            return float(abs(v))
    return float(abs(vals[-1]))


def figure_theory() -> None:
    """The accelerated rate against powers of r -- Figure 1 of the paper."""
    import matplotlib.pyplot as plt
    r = np.linspace(1e-3, 0.9999, 2000)
    rho = np.array([asymptotic_rate(v) for v in r])
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for a, powers, lo in ((ax[0], (1, 2, 3, 4, 6, 10), 0.0), (ax[1], (6, 10, 14, 20), 0.9)):
        m = r >= lo
        a.plot(r[m], rho[m], color="black", lw=2,
               label=r"$\rho(r)=r/(1+\sqrt{1-r^2})$")
        for pw in powers:
            a.plot(r[m], r[m] ** pw, lw=1.0, alpha=0.8, label=f"$r^{{{pw}}}$")
        a.set_xlabel(r"$r=|\lambda_2/\lambda_1|$")
        a.set_xlim(lo, 1.0)
        a.legend(ncol=2, fontsize=7.5)
    ax[0].set_ylabel("convergence rate")
    ax[0].set_title("How much momentum can win")
    ax[1].set_title(r"Detail near $r\to1$")
    ax[1].set_ylim(0, 0.45)
    save_figure(fig, "exp1_theory_rate")


def run_matrix(name: str, A, lambda2: float, r: float, rows: list, tol=TOL):
    res = run_all_methods(A, lambda2, tol=tol, max_iter=2000)
    print(f"  {name}:  r = {r:.4f}   "
          f"(theory predicts {predicted_speedup(r):.2f}x, rate {r:.4f} -> {asymptotic_rate(r):.4f})")
    for key, out in res.items():
        print(f"     {key:>8s}: {out.matvecs:5d} matvecs  residual {out.final_residual():.2e}  "
              f"converged={out.converged}")
        rows.append(dict(matrix=name, n=A.shape[0], r=r, method=key,
                         matvecs=out.matvecs, converged=out.converged,
                         residual=out.final_residual(), time_s=out.time,
                         predicted_speedup=predicted_speedup(r)))
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="skip SuiteSparse downloads")
    args = ap.parse_args()
    use_style()
    import matplotlib.pyplot as plt

    rows: list[dict] = []
    figure_theory()

    # ---------------- the paper's Matrix 1 ----------------
    banner("Matrix 1:  A = diag(1000 : -1 : 1)      (the paper's benchmark, r = 0.999)")
    A, ev = diag_benchmark(1000)
    res = run_matrix("diag(1000:-1:1)", A, ev[1], ev[1] / ev[0], rows)

    # The paper caps every run at 2000 iterations, so the power iteration never
    # finishes here and the printed speed-up would be meaningless.  Run it once
    # more uncapped to get its true cost.
    uncapped = power_iteration(A, tol=TOL, max_iter=100_000)
    sp_dyn = uncapped.matvecs / res["dynamic"].matvecs
    sp_sta = uncapped.matvecs / res["static"].matvecs
    print(f"     power (uncapped): {uncapped.matvecs} matvecs")
    print(f"     -> measured speed-up: static {sp_sta:.2f}x, dynamic {sp_dyn:.2f}x "
          f"(theory {predicted_speedup(0.999):.2f}x)")
    rows.append(dict(matrix="diag(1000:-1:1)", n=1000, r=0.999, method="power_uncapped",
                     matvecs=uncapped.matvecs, converged=uncapped.converged,
                     residual=uncapped.final_residual(), time_s=uncapped.time,
                     predicted_speedup=predicted_speedup(0.999)))

    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for key, out in res.items():
        y = np.asarray(out.residual)
        ax[0].semilogy(np.arange(1, y.size + 1), y, markevery=markevery(y.size), **STYLE[key])
    ax[0].set_xlabel("iteration"); ax[0].set_ylabel(r"$\|Ax_k-\nu_k x_k\|_2$")
    ax[0].set_title("Matrix 1: diag(1000:-1:1)"); ax[0].legend()
    b = np.asarray(res["dynamic"].beta)
    ax[1].plot(np.arange(1, b.size + 1), b, color=STYLE["dynamic"]["color"],
               label=r"$\beta_k$ chosen by Alg. 3.1")
    ax[1].axhline(optimal_beta(ev[1]), color="black", ls="--", lw=1.2,
                  label=r"$\beta_{opt}=\lambda_2^2/4$")
    ax[1].set_xlabel("iteration"); ax[1].set_ylabel(r"$\beta_k$")
    ax[1].set_title(r"$\beta_k$ finds the optimum unaided"); ax[1].legend()
    save_figure(fig, "exp1_matrix1")

    # ---------------- the paper's SuiteSparse matrices ----------------
    panels = []
    if not args.quick:
        banner("The paper's SuiteSparse benchmark matrices")
        for name in PAPER_MATRICES:
            try:
                M = load_suitesparse(name)
            except Exception as exc:                       # noqa: BLE001
                print(f"  {name}: unavailable ({type(exc).__name__}) -- skipped")
                continue
            vals = leading_eigenvalues(M)
            l1, l2 = abs(vals[0]), effective_lambda2(vals)
            r = l2 / l1
            print(f"  (paper reports r = {PAPER_R.get(name, float('nan')):.4f})")
            panels.append((name, r, run_matrix(name, M, l2, r, rows)))

        if panels:
            fig, axes = plt.subplots(1, len(panels), figsize=(4.1 * len(panels), 3.5),
                                     squeeze=False)
            for a, (name, r, res_m) in zip(axes[0], panels):
                for key, out in res_m.items():
                    y = np.asarray(out.residual)
                    a.semilogy(np.arange(1, y.size + 1), y,
                               markevery=markevery(y.size), **STYLE[key])
                a.set_xlabel("iteration"); a.set_title(f"{name}  ($r={r:.4f}$)")
            axes[0][0].set_ylabel(r"$\|Ax_k-\nu_k x_k\|_2$"); axes[0][0].legend()
            save_figure(fig, "exp1_suitesparse")

    df = pd.DataFrame(rows)
    df.to_csv(result_path("exp1_reproduce.csv"), index=False)
    print("\nwrote result/exp1_reproduce.csv")

    # ---------------- gate ----------------
    banner("GATE: is our implementation faithful to the paper?")
    checks = [
        ("dynamic momentum converges on Matrix 1", res["dynamic"].converged),
        (f"dynamic beats power by >5x ({sp_dyn:.1f}x measured)", sp_dyn > 5.0),
        (f"measured speed-up within 25% of the predicted {predicted_speedup(0.999):.1f}x",
         abs(sp_dyn - predicted_speedup(0.999)) < 0.25 * predicted_speedup(0.999)),
        ("dynamic is at least as fast as the optimal static method",
         res["dynamic"].matvecs <= res["static"].matvecs),
        (r"beta_k settles at lambda_2^2/4 without being told it",
         abs(np.median(res["dynamic"].beta[-50:]) - optimal_beta(ev[1]))
         < 0.05 * optimal_beta(ev[1])),
    ]
    for name, r_ in PAPER_R.items():
        got = df[(df.matrix == name)]["r"]
        if len(got):
            checks.append((f"measured r for {name} matches the paper ({r_})",
                           abs(got.iloc[0] - r_) < 5e-4))
    ok = True
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        ok &= bool(passed)
    print(f"\n  Gate: {'PASS -- the implementation is faithful' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
