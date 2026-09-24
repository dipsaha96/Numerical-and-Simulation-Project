# Accelerating PageRank with Dynamic Momentum Power Iteration

**CSE 402 — Numerical Analysis Sessional · Section C**

A methodological extension and cross-domain application of

> Austin, C., Pollock, S., & Zhu, Y. (2024). *Dynamically accelerating the power
> iteration with momentum.* Numerical Linear Algebra with Applications, **31**(6),
> e2584. [doi:10.1002/nla.2584](https://doi.org/10.1002/nla.2584) ·
> [arXiv:2403.09618v2](https://arxiv.org/abs/2403.09618)

| | |
|---|---|
| **Group members** | Nakib Arman (2105128) · Ali Asif Khan (2105131) · Shariar Al Kabir (2105132) · Dip Saha (2105138) · MD. Mehedi Hasan Mim (2105142) |
| **Presenter** | Dip Saha |
| **Language / stack** | Python 3.13, NumPy, SciPy, NetworkX (validation only), Matplotlib, pandas |
| **Code** | ~3,600 lines across 7 library modules, 7 experiment drivers, 19 unit tests |
| **Reproduce everything** | `make setup && make all` |

---

## Table of contents

1. [Executive summary](#1-executive-summary)
2. [Background: what the base paper does](#2-background-what-the-base-paper-does)
3. [Applying it to PageRank](#3-applying-it-to-pagerank)
4. [Implementation](#4-implementation)
5. [Results](#5-results)
6. [Why it fails: the spectral analysis](#6-why-it-fails-the-spectral-analysis)
7. [The safeguarded algorithm](#7-the-safeguarded-algorithm)
8. [Testing procedure](#8-testing-procedure)
9. [Reproducing every figure and table](#9-reproducing-every-figure-and-table)
10. [Limitations and threats to validity](#10-limitations-and-threats-to-validity)
11. [Work division](#11-work-division)
12. [References](#12-references)
13. [Appendix: file inventory and data schemas](#13-appendix-file-inventory-and-data-schemas)

---

## 1. Executive summary

The base paper proposes a *dynamic momentum* method (its Algorithm 3.1) that
accelerates the power iteration for the dominant eigenpair of a matrix. Its
momentum parameter `β_k` is re-estimated at every step from the Rayleigh
quotient and the ratio of the last two residuals, so — unlike earlier
heavy-ball methods — it needs no prior knowledge of `λ₂`, and it still costs
exactly **one matrix-vector product per iteration**. The paper benchmarks it
only on abstract SuiteSparse matrices and never applies it to a real ranking
problem.

PageRank is the obvious target: it *is* a dominant-eigenvector problem on a
huge sparse matrix, and it is slowest exactly when the two largest eigenvalues
are close — the regime the method is built for. This project implements the
method faithfully, applies it to real citation and web graphs, and reports what
happens.

**The headline result is negative, and the cause is specific and demonstrable.**

| # | Finding |
|---|---|
| **1** | **The method works exactly as advertised on the problems the paper covers.** On `diag(1000:-1:1)` we measure **40.4×** against a predicted 44.7×, and the dynamic method (683 matvecs) beats even the *optimal* static parameter (777). Our measured spectral ratios match every value the paper publishes. |
| **2** | **For the Google matrix the optimal static parameter is `β = d²/4`, exactly and for free.** `λ₁ = 1` by stochasticity, and Haveliwala–Kamvar gives `\|λ₂\| ≤ d`. We *measured* `\|λ₂(G)\| = d` to six decimals on `cit-HepPh` and `web-Stanford`. No eigensolve needed — a stronger baseline than the paper itself could use. |
| **3** | **Algorithm 3.1 as printed diverges on every real graph tested.** The residual oscillates; one step with `d_{k+1} > d_k` pins `ρ_k = 1`, hence `r_{k+1} = 1` and `β_k → λ₁²/4 = ¼` — precisely the value at which §2 of the paper proves convergence stops. |
| **4** | **The root cause is non-normality.** For real `λ`, every mode with `λ²/4 < β` maps to `\|μ\| = √β` exactly — that identity is what makes `β = λ₂²/4` optimal. For `λ = iy` it becomes `(\|y\| + √(y²+4β))/2`, unbounded in `\|y\|`. So the destabilising modes are **small-modulus eigenvalues near the imaginary axis**, which a largest-magnitude eigensolver never returns. |
| **5** | **A symmetric control isolates the cause.** On the *same* `cit-HepPh` data symmetrised as the co-citation matrix `AAᵀ`, the dynamic method delivers **3.58×** against 3.79× predicted. Only normality changed. |
| **5b** | **A single-variable sweep proves it.** Holding `λ₁ = 1` and `\|λ₂\| = 0.9` fixed and rotating *only* the argument of the subdominant eigenvalues: the power iteration varies 8–13% across the whole sweep while momentum flips from 3.14× to divergence after as little as a **3.6° rotation**. Theory predicts the onset for the static optimal parameter in **3/3** cases exactly. |
| **6** | **Our safeguarded variant restores robustness**, using only information PageRank supplies for free (`λ₁ = 1`, `r ≤ d`) plus residual-triggered backtracking. It converges on every graph and damping factor tested, reaching **4.08×** on `cit-HepPh` at `d = 0.99` where the published algorithm diverges. |
| **7** | **For ranking, none of it matters much.** The top-100 ordering is already correct after **7–16** matvecs, while driving the eigen-residual to `10⁻¹²` takes up to **2005**. Ranking converges **15–286×** sooner than the residual, so the acceleration is spent on digits nobody reads. |

---

## 2. Background: what the base paper does

### 2.1 The three iterations

All three cost one matrix-vector product per iteration. Notation follows the
paper exactly (`u`, `v`, `x`, `h`, `ν`, `d`, `ρ`, `r`) so the code can be read
side by side with Algorithms 1.1, 1.2 and 3.1.

**Algorithm 1.1 — power iteration.**

```
h_{k+1} = ‖v_{k+1}‖ ,  x_{k+1} = v_{k+1} / h_{k+1}
v_{k+2} = A x_{k+1}                      ← the single matvec
ν_{k+1} = ⟨v_{k+2}, x_{k+1}⟩             ← Rayleigh quotient
d_{k+1} = ‖v_{k+2} − ν_{k+1} x_{k+1}‖    ← residual
```

Converges at rate `r = |λ₂/λ₁|`.

**Algorithm 1.2 — static momentum** (heavy-ball, Xu–He–Gu 2018). One extra term:

```
u_{k+1} = v_{k+1} − (β / h_k) · x_{k−1}
```

**Algorithm 3.1 — dynamic momentum** (the paper's contribution):

```
Run 2 plain power iterations            → h₁,x₁,d₁, h₂,x₂,d₂, v₃
r₂ = min(d₂/d₁, 1)
for k ≥ 2:
    β_k     = ν_k² · r_k² / 4
    u_{k+1} = v_{k+1} − (β_k / h_k) · x_{k−1}
    h_{k+1} = ‖u_{k+1}‖ ;  x_{k+1} = u_{k+1}/h_{k+1}
    v_{k+2} = A x_{k+1}                  ← the only matvec
    ν_{k+1} = ⟨v_{k+2}, x_{k+1}⟩
    d_{k+1} = ‖v_{k+2} − ν_{k+1} x_{k+1}‖
    ρ_k     = min(d_{k+1}/d_k, 1)
    r_{k+1} = 2ρ_k / (1 + ρ_k²)
    stop if d_{k+1} < tol
```

### 2.2 The augmented matrix — the key analytical device

Iteration 1.2 is exactly the first `n` rows of a *plain power iteration* applied
to

```
A_β = ⎡ A   −βI ⎤
      ⎣ I    0  ⎦
```

whose eigenvalues are, for each eigenvalue `λ` of `A`,

```
μ± = ( λ ± √(λ² − 4β) ) / 2                                    ... (2.7)
```

So the momentum iteration converges at the rate
`max_{l≥2} |μ(λ_l)| / |μ(λ₁)|`. **Everything in this project follows from
equation (2.7).**

### 2.3 Why `β = λ₂²/4` is optimal — and the hidden assumption

For a **real** spectrum, every `λ` with `λ²/4 < β` has a *complex-conjugate*
`μ` pair of modulus **exactly `√β`**. Subdominant modes therefore can never be
worse than `√β`, and minimising over `β` gives `β_opt = λ₂²/4` with asymptotic
rate

```
ρ(r) = r / (1 + √(1 − r²)),        r = |λ₂/λ₁|                 ... (2.13)
```

Since reaching tolerance `ε` costs `log ε / log(rate)` iterations, the predicted
speedup over the power iteration is `log ρ(r) / log r`:

| r | ρ(r) | predicted speedup |
|---|---|---|
| 0.85 | 0.5567 | 3.60× |
| 0.90 | 0.6268 | 4.43× |
| 0.95 | 0.7239 | 6.30× |
| 0.99 | 0.8676 | 14.13× |
| 0.999 | 0.9562 | 44.72× |

The closer `r` is to 1, the bigger the win. **The hidden assumption is that the
spectrum is real.** The paper proves acceleration for symmetric `A`; §6 below
shows what happens when that fails.

### 2.4 How Algorithm 3.1 estimates `r` without an eigensolver

It inverts (2.13). If the *detected* residual ratio `ρ_k = d_{k+1}/d_k` is
behaving like the optimal rate, then

```
ρ = r / (1 + √(1−r²))     ⟺     r = 2ρ / (1 + ρ²)
```

Lemma 3.4 shows this inversion is *stable*, and increasingly so as `r → 1` — an
`ε` perturbation in `ρ` becomes `ε̂ < 2ε`, and `ε̂ < 0.047ε` for `r ∈ (0.999, 1)`.
This is elegant: the method is most reliable exactly where it is most needed.

---

## 3. Applying it to PageRank

### 3.1 The Google matrix

In the column-stochastic convention (`x` is a column of ranks):

```
G = d · ( P + (1/n) e aᵀ ) + ((1 − d)/n) e eᵀ
```

- `P[i,j] = 1/outdeg(j)` when `j → i`;
- `a_j = 1` exactly when node `j` is dangling (no out-links);
- the middle term patches dangling columns into uniform columns so `G` is
  stochastic; the last term is teleportation.

`G` is dense, so it is **never formed**. It is applied as

```
y = d·(P x) + (d/n)·(aᵀx)·e + ((1−d)/n)·(eᵀx)·e
```

— one sparse matvec plus two dot products, counted as **one matvec**.

### 3.2 Why PageRank looked like the ideal application

Three facts line up:

1. `λ₁(G) = 1` exactly, by column-stochasticity.
2. **Haveliwala & Kamvar (2003):** `|λ₂(G)| ≤ d` for every graph, with equality
   when the stochastic-completed link matrix has ≥2 irreducible closed subsets.
3. Therefore `r = |λ₂/λ₁| ≤ d`, and **`β_opt = λ₂²/4 = d²/4` needs no
   computation at all.**

This is strictly better than the paper's own situation: it had to call `eigs`
to get `λ₂` for its static baseline and stated plainly that this was "for
comparison purposes only". For PageRank the optimal parameter is a closed form
in a number the user already chose.

Whether the bound is *tight* is a property of the graph, so we measured it
rather than assuming (see §5.2). It is tight on the citation and web graphs and
slack on the social graph.

### 3.3 Two implementation traps

These cost real debugging time and are both regression-tested.

**Trap 1 — the `‖x‖₁ = 1` matvec shortcut.** Standard PageRank code writes

```python
y = d * (P @ x) + (d * a_dot_x + (1 - d)) / n      # WRONG under momentum
```

which is valid only because plain power iteration preserves `‖x‖₁ = 1`. The
momentum iterates are **ℓ₂-normalised and may contain negative entries**, so
`eᵀx` is neither one nor necessarily positive. Using the shortcut silently
produces a wrong operator and therefore a wrong ranking.
`test_google_matrix_preserves_total_mass` asserts `eᵀGx = eᵀx` for arbitrary
`x`, including negative and zero vectors.

**Trap 2 — capping `β_k` against `ν_k`.** Enforcing `β < λ₁²/4` by clipping to
`θ·ν_k²/4` looks harmless. It is not. Since `ν_k → λ₁`, any `θ < 1` clips `β_k`
below the optimum whenever `r² > θ` — exactly the near-degenerate regime the
method exists for. On `diag(1000:-1:1)` a **1% clip degrades the asymptotic
rate from 0.9562 to 0.9895 and quadruples the iteration count** (683 → ~2600).
The cap must be taken against a *known* `λ₁`, which PageRank provides.
`test_lambda1_cap_does_not_clip_the_optimum` guards this.

---

## 4. Implementation

### 4.1 Module map

```
src/
  iterations.py        Algorithms 1.1, 1.2, 3.1 + safeguards.  Matrix-agnostic.
  google_matrix.py     Implicit Google matrix operator; matvec counting.
  pagerank_methods.py  The five configurations compared in every experiment.
  spectrum.py          Augmented-matrix eigenvalues; true rate as a function of β.
  datasets.py          SNAP / SuiteSparse download + cache; synthetic fallback.
  metrics.py           ℓ1 error, Kendall tau, precision@k.
  plotting.py          Shared figure style.

experiments/
  exp1_reproduce.py    Phase 1 gate — reproduce the paper.
  exp2_pagerank.py     Phase 2 gate — validate G; then benchmark at d = 0.85.
  exp3_damping.py      Damping sweep; measured vs predicted speedup.
  exp4_ranking.py      Residual convergence vs ranking convergence.
  exp5_spectrum.py     Why β = d²/4 fails, with an empirical self-check.
  exp6_symmetric.py    Symmetric control.

tests/                 19 unit tests, including regressions for both traps.
results/               CSV output + run.log
figures/               PDF (for LaTeX) and PNG (for slides)
```

### 4.2 The one design decision that matters

**`src/iterations.py` never imports anything about PageRank.** Every routine
takes an arbitrary linear operator — a dense array, a SciPy sparse matrix, or a
`LinearOperator` — and only ever computes `A @ x`.

This is what makes the SuiteSparse reproduction (§5.1) and the Google-matrix
study (§5.3) share **one code path**, and it is what makes the symmetric
control (§5.6) free: the same three solvers run on `AAᵀ` with no changes. When
the PageRank results came out negative, that shared code path is what let us
prove the solvers were correct rather than argue about it.

### 4.3 The five configurations compared

Defined once in `src/pagerank_methods.py` so every experiment compares exactly
the same thing:

| name | what it is |
|---|---|
| `power` | Algorithm 1.1. The baseline every speedup is measured against. |
| `static_d` | Algorithm 1.2 with `β = d²/4` — the paper's optimal parameter, evaluated exactly and for free (§3.2). |
| `dynamic_paper` | Algorithm 3.1 exactly as printed. |
| `dynamic_safe` | Algorithm 3.1 + the `λ₁ = 1` cap + the `r ≤ d` bound + backtracking with slow growth. |
| `dynamic_guard` | `dynamic_safe` + the rate guard (retreats as soon as momentum stops paying). |

### 4.4 Early-stopping rules

A failing run must not consume the whole budget, and the output must say *how*
it failed. Two rules, applied only where backtracking is not available to
recover:

- **`divergence_factor` (default `10³`)** — stop once the residual exceeds its
  best value by that factor.
- **`stall_window` (default 1000)** — stop after that many iterations with no
  new best residual. This is the failure mode that actually occurs: with `β`
  past the stability boundary the residual does not blow up, it *plateaus*.

Neither can fire on a converging run, since convergence means the best residual
keeps improving. Adding them cut `static_d` on `web-Stanford` from 80,001
matvecs / **467 s** to 1,014 matvecs / **3.6 s**, reporting `diverged=True`.

---

## 5. Results

All counts are **matrix-vector products** to residual `10⁻¹²`, started from
`u₀ = (1,…,1)ᵀ`. `!` marks a run stopped early as diverged/stalled; such runs
are reported with **no** speedup, because a method that gave up after 1,004 of
a possible 80,000 matvecs has not "won".

### 5.1 Phase 1 — reproducing the base paper

| matrix | n | r = \|λ₂/λ₁\| | power | static `β=λ₂²/4` | **dynamic (Alg. 3.1)** |
|---|---|---|---|---|---|
| `diag(1000:-1:1)` | 1,000 | 0.9990 | 27,619 | 777 | **683** |
| `Kuu` | 7,102 | 0.9981 | >2,000 | 1,166 | **528** |
| `Muu` | 7,102 | 0.9992 | >2,000 | 728 | **292** |
| `ash292` | 292 | 0.9153 | 313 | 75 | **73** |
| `bcspwr06` | 1,454 | 0.9814 | 1,360 | 154 | **145** |

Every measured spectral ratio matches the paper's published value. On
`diag(1000:-1:1)` the measured speedup is **40.4×** against the predicted
44.7×, and **the dynamic method beats the optimal static parameter on all
five matrices** — the paper's central claim, reproduced.

`Muu` is a nice detail: it has `λ₂ = λ₁` to machine precision, and the paper
notes Algorithm 1.2 fails there and substitutes `λ₃` by hand. Our
`effective_lambda2` helper detects the degeneracy automatically, and — as the
paper says — Algorithm 3.1 needs no modification at all.

**Initial-vector sensitivity (Table 1 of the paper), 30 random starts:**

| matrix | r | method | mean | min | max | std |
|---|---|---|---|---|---|---|
| `diag(200:-1:1)` | 0.995 | static | 336.4 | 326 | 358 | 8.2 |
| | | dynamic | **309.0** | 287 | 335 | 9.9 |
| `diag(linspace(−99,100,200))` | 0.990 | static | **262.7** | 240 | 306 | 11.8 |
| | | dynamic | 387.3 | 253 | 889 | **117.8** |

This reproduces the paper's own caveat: on *indefinite* matrices the dynamic
method is markedly **more** sensitive to the starting vector than the static
one.

### 5.2 Phase 2 — is `|λ₂(G)| = d`?

Measured with ARPACK on the implicit operator:

| graph | n | dangling | `\|λ₂(G)\|` at d=0.85 | `λ₂/d` | verdict |
|---|---|---|---|---|---|
| `wiki-Vote` | 7,115 | 1,005 (14.1%) | 0.501427 | 0.5899 | **slack** — `d²/4` overshoots |
| `cit-HepPh` | 34,546 | 2,393 (6.9%) | 0.850000 | 1.0000 | **tight** — `d²/4` exactly optimal |
| `web-Stanford` | 281,903 | 172 (0.1%) | 0.850000 | 1.0000 | **tight** |

The Haveliwala–Kamvar bound is tight on the citation and web graphs — exactly
the domain the proposal targets — and slack on the social voting graph, which
is strongly connected enough to have a single closed subset. `wiki-Vote` is
therefore a useful counterexample, not a failure.

### 5.3 Phase 3 — PageRank benchmark at `d = 0.85`

| graph | power | `static_d` | `dynamic_paper` | `dynamic_safe` | `dynamic_guard` |
|---|---|---|---|---|---|
| `wiki-Vote` | **39** | 78 (0.50×) | 1,018 ! | 68 (0.57×) | 68 (0.57×) |
| `cit-HepPh` | **126** | 162 (0.78×) | 1,005 ! | 153 (0.82×) | 153 (0.82×) |
| `web-Stanford` | **152** | 1,014 ! | 1,006 ! | 362 (0.42×) | 312 (0.49×) |

**Nothing beats the plain power iteration.** Algorithm 3.1 as printed diverges
on all three. All converging methods agree on the answer to `ℓ1 ≈ 5×10⁻¹³` and
`precision@100 = 1.000`, so the comparison is honest — they are solving the
same problem, just not faster.

### 5.4 Phase 4 — the damping sweep

`cit-HepPh` (`λ₂ = d` exactly, so the theory's premise holds):

| d | power | `static_d` | `dynamic_paper` | `dynamic_safe` | speedup | **theory** |
|---|---|---|---|---|---|---|
| 0.85 | 126 | 162 | 1,005 ! | 153 | 0.82× | 3.60× |
| 0.90 | 193 | 526 | 1,005 ! | 491 | 0.39× | 4.43× |
| 0.95 | 395 | 1,004 ! | 1,005 ! | **173** | **2.28×** | 6.30× |
| 0.99 | 2,005 | 1,004 ! | 1,005 ! | **492** | **4.08×** | 14.13× |
| 0.999 | 19,665 | 1,004 ! | 1,005 ! | 17,568 | 1.12× | 44.72× |

`web-Stanford` never exceeds parity (0.42×–0.70×). `wiki-Vote` peaks at 1.10×.
The safeguarded method does deliver real acceleration on `cit-HepPh` in the
`d ∈ [0.95, 0.99]` band, but always far below the predicted figure, and the
published algorithm diverges everywhere.

### 5.5 Phase 5 — ranking convergence vs residual convergence

`cit-HepPh`, `k = 100`, measured against a reference computed to `10⁻¹⁴`:

| d | method | matvecs to residual `10⁻¹²` | matvecs to `precision@100 = 1.0` | matvecs to `τ > 0.99` | ratio |
|---|---|---|---|---|---|
| 0.85 | power | 126 | **8** | 7 | 15.8× |
| 0.85 | `dynamic_safe` | 153 | 17 | 21 | 9.0× |
| 0.99 | power | 2,005 | **7** | 9 | **286.4×** |
| 0.99 | `dynamic_safe` | 492 | 16 | 17 | 30.8× |

This is the practically important result and the one the base paper's
benchmarks cannot produce at all. **The top-100 ranking is correct after 7–16
matrix-vector products.** Everything after that is refining digits that no
ranking application reads. For PageRank specifically, accelerating the tail of
the eigen-residual optimises the wrong quantity.

### 5.6 Phase 6 — the symmetric control

Every negative result needs a control, or it cannot be distinguished from a
bug. These are genuine ranking problems whose matrices are symmetric, so
Theorem 3.1 applies and the real-spectrum rate is the truth:

| problem | n | r | power | static | **dynamic** | **predicted** |
|---|---|---|---|---|---|---|
| `ca-HepPh` eigenvector centrality | 12,008 | 0.3751 | 35 | 23 (1.52×) | **23 (1.52×)** | 1.67× |
| `cit-HepPh` co-citation `AAᵀ` | 34,546 | 0.8638 | 236 | 72 (3.28×) | **66 (3.58×)** | 3.79× |

The second row is the decisive one: **the same citation data**, symmetrised,
and the acceleration appears at 94% of the predicted value. The only thing that
changed is normality.

### 5.7 Phase 8 — isolating non-normality as the single cause

The control in §5.6 has a gap a sharp reader will find: relative to the PageRank
case it changes **three** things at once — the symmetry, the matrix itself, and
`r`. Observing a different outcome does not establish *which* one is
responsible.

This experiment closes the gap. It builds a family of matrices with `λ₁ = 1`
and `|λ₂| = r` **fixed** — so the predicted speedup is identical for every
member — and varies only the **argument** of the subdominant eigenvalues, from
`0` (real spectrum, symmetric matrix) to `π/2` (purely imaginary). Each matrix
is block diagonal with `[1]` in the corner and 2×2 rotation-scaling blocks whose
eigenvalues are `m·e^{±iθ}`.

Because `|λ₂|` never changes, the **plain power iteration is an internal
control**: it should be equally hard throughout.

| arg(λ)/π | symmetric? | power | static `β=r²/4` | Alg. 3.1 | + safeguards | speedup |
|---|---|---|---|---|---|---|
| 0.00 | yes | 245 | 64 | **78** | 68 | **3.14×** |
| 0.02 | no | 247 | 120 | 1022 ! | 121 | **diverged** |
| 0.05 | no | 251 | 322 | 1017 ! | 317 | diverged |
| 0.10 | no | 256 | 1010 ! | 1011 ! | 406 | diverged |
| 0.50 | no | 270 | 1002 ! | 1003 ! | 592 | diverged |

*(`r = 0.9`, predicted 4.43× on every row; `!` = did not converge)*

**The power iteration moves by 8–13% across the entire sweep** while momentum
flips from 3.14× to divergence after as little as a **3.6° rotation** — the
smallest angle we sample. One variable in, one
outcome out.

**The sweep also separates the two failure modes of §6**, which the PageRank
experiments could only observe tangled together:

| | mechanism | onset |
|---|---|---|
| **Mode B** | the spectral crossing — `max\|μ(λ_l)\|` overtakes `\|μ(λ₁)\|` | kills the *statically optimal* `β = r²/4` |
| **Mode A** | the `ρ_k → 1` runaway | kills **Algorithm 3.1 earlier**, at the first hint of oscillation |

Mode B is predicted **exactly**:

| r | crossing after | theory predicts divergence at | measured | |
|---|---|---|---|---|
| 0.85 | 0.10π | 0.15π | 0.15π | ✅ |
| 0.90 | 0.05π | 0.10π | 0.10π | ✅ |
| 0.95 | 0.02π | 0.05π | 0.05π | ✅ |

**3/3 exact agreement** between the predicted crossing and the measured onset —
a quantitative validation of §6, not merely a qualitative one. Algorithm 3.1
fails *before* the crossing because it infers `r` from residual ratios: one step
with `d_{k+1} > d_k` sets `ρ_k = 1`, hence `r = 1` and `β → ¼`. **Its
self-tuning rule steers it into the barrier that the fixed parameter still
avoids.**

The safeguarded variant converged on **27/27** of the same matrices.

---

## 6. Why it fails: the spectral analysis

This is the project's main analytical contribution — the mechanism behind
findings 3 and 4.

### 6.1 Failure mode A — the `ρ_k = 1` runaway

The Google matrix is non-normal, its residual oscillates, and Algorithm 3.1's
`min(ρ_k, 1)` clamp — inert on a symmetric matrix where the residual falls
monotonically — becomes active. A single step with `d_{k+1} > d_k` gives
`ρ_k = 1`, hence

```
r_{k+1} = 2·1/(1+1) = 1        ⟹     β_{k+1} = ν²/4 → λ₁²/4 = ¼
```

which §2 of the paper proves is exactly where every eigenvalue of `A_β` has the
same magnitude `√β` and convergence stops. The residual then grows, `ρ_k` stays
pinned at 1, and the method never recovers.

Measured on `wiki-Vote` at `d = 0.85`: the residual bottoms out near `10⁻³`
around iteration 20, then climbs back to `1.5×10⁻¹`. **2,000 iterations end
further from the answer than 20 did.**

### 6.2 Failure mode B — complex eigenvalues (the deeper cause)

Bounding `r ≤ d` fixes A but not this. Return to (2.7). For a **real** `λ` with
`λ²/4 < β`, `|μ±| = √β` exactly. For `λ = iy`:

```
μ± = i( y ± √(y² + 4β) ) / 2       ⟹     |μ| = ( |y| + √(y² + 4β) ) / 2
```

which **exceeds `√β` for every `y ≠ 0` and grows without bound in `|y|`.**

The consequence inverts the intuition the real case builds: the destabilising
modes are **not** near the top of the spectrum but *small-modulus* eigenvalues
lying near the imaginary axis — which a largest-magnitude eigensolver never
even returns. `src/spectrum.py` therefore samples `which="LM"`, `"LI"` **and**
`"SI"`.

**Measured, `wiki-Vote`, `d = 0.99`, `β = d²/4 = 0.245`:**

| λ | \|λ\| | arg(λ)/π | \|μ(λ)\| | |
|---|---|---|---|---|
| `+0.0102 + 0.2312i` | **0.2314** | 0.486 | **0.6239** | ← exceeds `\|μ_λ₁\|` |
| `−0.0351 + 0.2307i` | 0.2333 | 0.548 | 0.6237 | ← exceeds |
| `λ₁ = 1` | 1.0000 | 0.000 | 0.5705 | dominant mode |

Eigenvalues of modulus **0.23** reach `|μ| = 0.624` against `|μ_λ₁| = 0.571`,
so the augmented iteration converges to **the wrong eigenmode**. 126 of the 179
sampled modes overtake the dominant one.

See `figures/exp5_spectrum_plane.png`: every diverging mode (circled red) sits
off the real axis at small modulus.

### 6.3 The true optimal `β` for a non-normal matrix

`exp5` computes the real rate `max|μ(λ)| / |μ(λ₁)|` over the sampled spectrum
as a function of `β`:

| graph | d | `β = d²/4` | true rate there | `β_best` | rate at `β_best` |
|---|---|---|---|---|---|
| `cit-HepPh` | 0.85 | 0.1806 | 0.8436 | 0.1472 | 0.7407 |
| `cit-HepPh` | 0.95 | 0.2256 | **1.0970 → diverges** | 0.1838 | 0.8973 |
| `cit-HepPh` | 0.99 | 0.2450 | **1.3147 → diverges** | 0.1996 | 0.9774 |
| `wiki-Vote` | 0.99 | 0.2450 | **1.1003 → diverges** | 0.0712 | 0.4451 |

**The paper's optimal parameter has a true convergence rate above 1.** It is
not merely suboptimal for PageRank; it is outside the stability region.

### 6.4 How little momentum is provably safe

Without spectral information, the worst case over the disk `|λ| ≤ d` is a
purely imaginary eigenvalue `λ = id`. Solving
`d + √(d²+4β) = 1 + √(1−4β)` gives the largest universally safe `β`:

| d | `β = d²/4` | provably safe `β` | ratio |
|---|---|---|---|
| 0.85 | 0.180625 | 0.069375 | 2.6× too large |
| 0.90 | 0.202500 | 0.047500 | 4.3× too large |
| 0.95 | 0.225625 | 0.024375 | 9.3× too large |
| 0.99 | 0.245025 | 0.004975 | **49× too large** |

As `d → 1` the safe momentum collapses toward zero — i.e. toward the plain
power iteration. This is the formal statement of why the approach cannot
deliver its promise on PageRank.

### 6.5 Honesty about the sampling

The rates above come from a 179-eigenvalue *sample* of a spectrum with up to
281,903 eigenvalues. Since the true rate is a **maximum** over the whole
spectrum, a sampled rate can only be **too low** — so:

- a sampled rate **above 1 proves divergence** (sound);
- a sampled rate below 1 does **not** prove convergence (not complete);
- the "achievable speedup" column is **optimistic**.

Rather than leave this as a caveat, `exp5` runs the static method at its own
recommended `β_best` and prints the measured result beside the prediction:

| graph | d | predicted from sample | **measured** | verdict printed |
|---|---|---|---|---|
| `cit-HepPh` | 0.85 | 1.85× | 1.26× | SAMPLE INCOMPLETE |
| `cit-HepPh` | 0.99 | 2.27× | 1.61× | sample adequate |
| `wiki-Vote` | 0.99 | 80.54× | **1.41×** | SAMPLE INCOMPLETE |

**Use the measured column, never the predicted one.** The 80× figure is an
artifact of incomplete sampling and the script says so itself.

---

## 7. The safeguarded algorithm

Our extension. It uses only information PageRank supplies for free, plus
feedback from the residual.

| safeguard | what it uses | what it fixes |
|---|---|---|
| **`λ₁` cap** — `β_k ≤ θ·λ₁²/4`, `θ = 1−10⁻⁸` | `λ₁ = 1` exactly, by stochasticity | Keeps `β` strictly inside the convergence region. Taken against a *known* `λ₁`, never against `ν_k` (Trap 2). |
| **`r_max` bound** — `r_k ← min(r_k, d)` | Haveliwala–Kamvar `\|λ₂\| ≤ d` | Kills failure mode A. Always valid, never excludes the true `r`. |
| **Backtracking** — after `W = 8` iterations with no new best, shrink the ceiling ×0.5 and restart from the best iterate | nothing — pure residual feedback | Handles failure mode B, which no a priori bound can prevent. |
| **Slow growth** — after 4 consecutive improvements, raise the ceiling ×1.06, capped at `r_max` | nothing | Retreat alone overshoots downward. Alternating shrink/grow makes the ceiling hover just below the stability boundary. |
| **Backtrack budget** — 20, then momentum off permanently | nothing | Turns the safeguard into a **guarantee**: worst case is the power iteration plus `3×20` matvecs. Without it an unbudgeted version spent 7,240 backtracks on `cit-HepPh` at `d = 0.999` and never converged. |
| **Rate guard** (optional, off by default) | priming residual ratio | Retreats as soon as momentum underperforms. Never materially worse than power iteration (0.94×–1.00×), but gives up the acceleration too. |

Two subtle bugs found while building this, both now regression-tested:

- **Once the ceiling reaches zero, backtracking must switch off.** Otherwise the
  method re-primes from the same best iterate forever, burning three matvecs a
  time. (`test_backtracking_terminates_without_momentum`)
- **The stagnation counter must advance even when backtracking is disabled**,
  or the stall detector never fires and a stalled Algorithm 3.1 runs out the
  full budget.

---

## 8. Testing procedure

Verification is layered. Each level is independently runnable and each one
either passes or tells you exactly what broke.

### 8.0 Prerequisites and setup

Requires Python 3.13+ and ~90 MB of disk for cached datasets. No GPU, no MPI.

```bash
cd "Numerical & Simulation Project"
make setup          # creates .venv and installs pinned dependencies
```

Pinned in `requirements.txt`: numpy 2.5.3, scipy 1.18.1, matplotlib 3.11.2,
pandas 3.0.6, networkx 3.6.1, ssgetpy 1.0rc2, pytest 9.1.1.

> **No network?** Every experiment accepts `--offline` and falls back to a
> synthetic generator with power-law out-degrees and dangling nodes. Run
> `make quick`. The pipeline exercises the same code paths; only the graphs
> differ.

### 8.1 Level 0 — unit tests (~1 s)

```bash
make test           # or: ./.venv/bin/python -m pytest tests/ -q
```

Expected: `19 passed`.

**`tests/test_iterations.py` — solver correctness (10 tests)**

| test | what it proves |
|---|---|
| `test_power_finds_dominant_eigenpair` | Power iteration returns `(λ₁, φ₁)` on a matrix with known spectrum. |
| `test_momentum_methods_agree_with_power` | All three methods return the same eigenpair (up to sign) to `10⁻⁶`. |
| `test_one_matvec_per_iteration` | Each method uses ≤3 matvecs more than its iteration count. **This is the premise of the whole comparison** — reporting iteration counts is only fair if per-iteration cost is identical. |
| `test_momentum_accelerates_symmetric_problem` | Theorem 3.1: >5× speedup on `diag(500:-1:1)`. |
| `test_dynamic_matches_predicted_rate` | Measured speedup tracks `log ρ(r)/log r` within 35%. |
| `test_dynamic_beta_converges_to_optimal` | `β_k → λ₂²/4` within 5% **without being told `λ₂`**. |
| `test_lambda1_cap_does_not_clip_the_optimum` | **Regression for Trap 2.** The cap must not fire on a well-posed symmetric problem. |
| `test_rate_formulas` | `ρ(0.85) = 0.5567262`, `speedup(0.999) = 44.72`, and `ρ(r) < r` strictly on `(0,1)`. |
| `test_nonsymmetric_needs_safeguards` | Reproduces failure mode B in **8 dimensions** using rotation blocks with eigenvalues `±0.9i`: asserts Algorithm 3.1 fails and the safeguarded variant recovers. |
| `test_backtracking_terminates_without_momentum` | **Regression** for the infinite re-priming bug. |

**`tests/test_google_matrix.py` — operator correctness (9 tests)**

| test | what it proves |
|---|---|
| `test_link_matrix_is_column_stochastic` | Non-dangling columns sum to 1; dangling columns are all-zero. |
| `test_google_matrix_preserves_total_mass` | **Regression for Trap 1.** `eᵀGx = eᵀx` for ones, random, **negative**, and zero vectors. |
| `test_pagerank_is_the_fixed_point` | Result is strictly positive (Perron–Frobenius), sums to 1, and `‖Gx−x‖₁ < 10⁻¹²`. |
| `test_dominant_eigenvalue_is_one` | `ν → 1` to `10⁻⁹`. |
| `test_matvec_handles_complex_input` | ARPACK probes with complex vectors; silently casting to float would corrupt §6's diagnostics. |
| `test_with_damping_reuses_structure` | `with_damping` shares the sparse structure rather than rebuilding. |
| `test_self_loops_and_duplicates_are_dropped` | Raw SNAP dumps contain both. |
| `test_normalise_ranking_fixes_sign` | Momentum can return `−φ₁`; normalisation must flip it. |
| `test_ranking_metrics_are_exact_on_identical_input` | Metrics are calibrated: identical input gives `ℓ1 = 0`, `precision = 1`, `τ = 1`. |

### 8.2 Level 1 — the Phase 1 gate (~60 s, downloads 4 SuiteSparse matrices)

**This gate must pass before any PageRank result means anything.** If our
Algorithm 3.1 does not behave the way the paper says it does on the paper's own
benchmarks, the implementation is wrong.

```bash
make phase1         # ./.venv/bin/python experiments/exp1_reproduce.py
```

Expected tail:

```
  [PASS] dynamic momentum converges on Matrix 1
  [PASS] dynamic beats plain power by >5x (40.4x measured)
  [PASS] measured speedup is within 25% of the predicted 44.7x
  [PASS] dynamic is at least as fast as the optimal static method
  [PASS] dynamic beta_k settles near beta_opt

  Phase 1 gate: PASS -- proceed to Phase 2
```

The script also prints measured `r` for each SuiteSparse matrix next to the
paper's published value — an independent check that we built the right
matrices. All four agree to four decimals.

Exit code is **0 on PASS, 1 on FAIL**, so it can be wired into CI.

### 8.3 Level 2 — the Phase 2 gate (~4 min, downloads ~90 MB)

Validates the Google matrix two independent ways.

```bash
make phase2
```

Expected:

```
  [PASS] matches networkx.pagerank (l1 error 3.559e-10, tolerance 1.4e-08);
         fixed-point residual ||Gx-x||_1 = 1.40e-15
  [ OK ] |lambda_2(G)| = 0.501427, d = 0.85 -> lambda_2/d = 0.5899  (bound SLACK)
  ...
  Phase 2 gate: PASS
```

Two checks, because neither alone is conclusive:

1. **Agreement with `networkx.pagerank`.** The tolerance *scales with n*:
   NetworkX stops when the per-iteration `ℓ1` change falls below `n·tol`, so it
   is the **less** accurate of the two and agreement can be no better than its
   own stopping accuracy. At `n = 34,546` that floor is `3.5×10⁻⁹`, and the
   measured disagreement of `1.8×10⁻⁸` is consistent with NetworkX carrying the
   error. Skipped above 100,000 nodes, where NetworkX takes tens of minutes.
2. **The fixed-point residual `‖Gx − x‖₁`** — self-contained, independent of any
   other library, applied at **every** size, costing one matvec. Measured
   `10⁻¹⁵` on all three graphs.

### 8.4 Level 3 — the control experiment (~15 s)

```bash
make phase6
```

This is the test that makes the negative result trustworthy. Expected:

```
  Control: momentum delivers the predicted acceleration on symmetric problems: PASS
```

If §5.3 showed no speedup *and* this showed no speedup, the honest conclusion
would be "our code is broken". It shows 3.58× against 3.79× predicted on the
same citation data, so the conclusion is "the Google matrix is the problem".

### 8.4b Level 3b — the single-variable sweep (~20 s, no downloads)

```bash
make phase7
```

Where §8.4 shows momentum works on *some* symmetric problem, this shows it stops
working the moment — and only the moment — the spectrum leaves the real axis,
with every other quantity pinned. Expected tail:

```
    -> 3/3 exact agreement between the predicted
       crossing and the measured onset.
  Safeguarded variant converged on 27/27 of the same matrices.
  One variable changed; one outcome flipped.  Non-normality is the cause.
```

This is the strongest single piece of evidence in the project and the cheapest
to re-run: pure synthetic matrices, no network, about 20 seconds.

### 8.5 Full pipeline

```bash
make all            # tests + all six experiments, ~15 min after caching
```

or equivalently `./run_all.sh`, which additionally tees clean output to
`results/run.log` (stderr is discarded — it carries only progress bars).

| phase | script | approx. runtime | produces |
|---|---|---|---|
| 0 | `pytest` | 1 s | — |
| 1 | `exp1_reproduce.py` | 60 s | `fig1_rate_comparison`, `fig3_matrix1`, `exp1_suitesparse` |
| 2–3 | `exp2_pagerank.py` | 4 min | `exp2_pagerank_convergence` |
| 4 | `exp3_damping.py` | 7 min | `exp3_damping_sweep` |
| 5 | `exp4_ranking.py` | 60 s | `exp4_ranking_quality` |
| 6 | `exp5_spectrum.py` | 90 s | `exp5_rate_vs_beta`, `exp5_spectrum_plane` |
| 7 | `exp6_symmetric.py` | 15 s | `exp6_symmetric_control` |
| 8 | `exp7_normality.py` | 20 s | `exp7_normality` |

First run adds ~90 MB of downloads. Every script accepts `--graphs`,
`--dampings` and `--offline`.

### 8.6 Determinism check

ARPACK starts from a **random** vector unless one is supplied, which feeds
straight into `β = λ₂²/4` and moves the static method's count by a percent or
two between runs. All eigensolves therefore pass `v0=np.ones(n)`. To verify:

```bash
./.venv/bin/python experiments/exp6_symmetric.py > /dev/null
cp results/exp6_symmetric.csv /tmp/a.csv
./.venv/bin/python experiments/exp6_symmetric.py > /dev/null

./.venv/bin/python -c "
import pandas as pd
a = pd.read_csv('/tmp/a.csv'); b = pd.read_csv('results/exp6_symmetric.csv')
cols = [c for c in a.columns if c != 'time_s']   # wall-clock is not deterministic
print('REPRODUCIBLE' if a[cols].equals(b[cols]) else 'MISMATCH')"
```

Expected: `REPRODUCIBLE`.

Compare every column **except `time_s`**. Wall-clock timing varies by a few
percent between runs on any machine; a plain `diff` of the two CSVs will always
report a difference on that column alone and says nothing about correctness.
The quantities the conclusions rest on — `matvecs`, `residual`, `speedup`, `r` —
are bit-identical.

### 8.7 Verifying the headline claims by hand

Each of these is a self-contained snippet a reviewer can paste to check one
claim without trusting the experiment drivers.

**Claim: the method achieves ~40× on the paper's benchmark.**

```python
from src.iterations import power_iteration, dynamic_momentum, predicted_speedup
from src.datasets import diag_benchmark
A, vals = diag_benchmark(1000)
p = power_iteration(A, tol=1e-12, max_iter=100_000)
d = dynamic_momentum(A, tol=1e-12, max_iter=100_000)
print(p.matvecs / d.matvecs, "vs predicted", predicted_speedup(999/1000))
# 40.4 vs predicted 44.7
```

**Claim: `|λ₂(G)| = d` on a citation graph.**

```python
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs
from src.datasets import load_snap_edges
from src.google_matrix import build_google_matrix
G, _ = build_google_matrix(*load_snap_edges("cit-HepPh"), d=0.85)
op = LinearOperator((G.n, G.n), matvec=G.matvec, dtype=float)
v = np.asarray(eigs(op, k=8, which="LM", v0=np.ones(G.n), tol=1e-6,
                    return_eigenvectors=False))
print(sorted(abs(v))[-2])      # 0.85
```

**Claim: `β = d²/4` is outside the stability region.**

```python
from src.spectrum import sample_spectrum, convergence_rate
spec = sample_spectrum(G.with_damping(0.95), G.n, k=60)
print(convergence_rate(spec, 0.95**2/4))     # 1.0970  -> > 1, diverges
```

**Claim: the ranking is right long before the residual is small.**

```python
from src.metrics import precision_at_k
from src.google_matrix import normalise_ranking
H = G.with_damping(0.99)
ref = normalise_ranking(power_iteration(H, tol=1e-14, max_iter=100_000).x)
run = power_iteration(H, tol=1e-12, max_iter=100_000, track_iterates=True)
first = next(i for i, x in enumerate(run.iterates, 1)
             if precision_at_k(x, ref, 100) == 1.0)
print(first, "vs", run.matvecs)      # 7 vs 2005
```

### 8.8 Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `ModuleNotFoundError: numpy` | venv not created | `make setup` |
| Downloads hang or fail | SNAP/SuiteSparse unreachable | Add `--offline`, or `make quick` |
| `Kuu: unavailable (FileNotFoundError)` | `ssgetpy` reports the tarball, not the extracted dir | Already handled — we glob `data/<name>/**/*.mtx` ourselves |
| A run sits at 100% CPU for minutes | ARPACK on a clustered spectrum, or a diverging run | Expected; `stall_window` bounds it. `web-Stanford` eigs legitimately takes ~2 min |
| Speedup column shows `NaN` | The run diverged | Intended — a method that gave up has no meaningful speedup |
| Figures look empty | Matplotlib backend | `plotting.py` forces `Agg`; check `figures/` for the PDFs |

---

## 9. Reproducing every figure and table

| artifact | produced by | shows |
|---|---|---|
| `fig1_rate_comparison` | `exp1` | Reproduction of the paper's Figure 1: `ρ(r)` vs `rᵖ`. |
| `fig3_matrix1` | `exp1` | Paper's Figure 3 (left) + the `β_k` trace finding `β_opt` unaided. |
| `exp1_suitesparse` | `exp1` | Convergence on `Kuu`, `Muu`, `ash292`, `bcspwr06`. |
| `exp2_pagerank_convergence` | `exp2` | Per-graph residual histories + `β_k` against `d²/4` and the `¼` barrier. |
| `exp3_damping_sweep` | `exp3` | Measured vs predicted speedup; cost as the spectral gap closes. |
| `exp4_ranking_quality` | `exp4` | Eigen-residual beside `precision@100`. The gap is the point. |
| `exp5_rate_vs_beta` | `exp5` | True rate vs `β`, with `d²/4` marked past the stability boundary. |
| `exp5_spectrum_plane` | `exp5` | **The key figure.** Spectrum in ℂ coloured by `\|μ(λ)\|`; diverging modes circled. |
| `exp6_symmetric_control` | `exp6` | The control: acceleration appears on symmetric problems. |
| `exp7_normality` | `exp7` | **The single-variable proof.** A: speedup vs `arg λ` with `r` pinned. B: the two failure modes. C: theory predicting mode B's onset exactly. |

All figures are written as **both PDF** (for LaTeX) **and PNG** (for slides).
Every table in §5 comes from the corresponding `results/*.csv`.

---

## 10. Limitations and threats to validity

Stated plainly, since the conclusion is negative.

1. **Spectral sampling is incomplete** (§6.5). Sampled rates are lower bounds,
   so divergence claims are sound but "achievable speedup" figures are
   optimistic. `exp5` measures its own predictions and flags the gap.
2. **Three graphs, one graph family each.** Partly mitigated by §5.7, which
   establishes the mechanism on a controlled synthetic family rather than
   relying on the graph sample. `cit-HepPh` (citation),
   `web-Stanford` (web), `wiki-Vote` (social). `web-Google` (876k) and
   `web-BerkStan` are supported but untested here.
3. **The safeguard parameters were tuned on these graphs.** `W = 8`,
   grow 1.06, budget 20 are reasonable but not derived from theory. The
   *guarantee* (never much worse than power iteration) does not depend on them;
   the *acceleration* does.
4. **Wall-clock is reported but iteration counts are the fair metric.** All
   methods are one matvec per iteration, so matvec counts are
   implementation-independent; timings reflect our Python/SciPy stack.
5. **We did not implement DMPower** (Algorithm 1 of the delayed-momentum
   paper), which the base paper also compares against. It costs 3 matvecs per
   iteration in its first phase and was out of scope; its absence does not
   affect any conclusion here, since the winner throughout is the plain power
   iteration.
6. **The negative result is about the Google matrix, not about the method.**
   §5.1 and §5.6 show the method works. The claim is specifically that
   non-normality with near-imaginary-axis eigenvalues defeats it.

---

## 11. Work division

| area | owner(s) | deliverable |
|---|---|---|
| Numerics | 2 members | `src/iterations.py`, Phase 1 reproduction, safeguard design and analysis |
| Graph & data | 1 member | `src/google_matrix.py`, `src/datasets.py`, dangling-node handling, Phase 2 gate |
| Experiments | 1 member | The six drivers, figures, results tables |
| Analysis & writing | 1 member (+ Dip presenting) | Spectral theory (§6), this document, slides |

The `history` schema returned by every solver
(`residual`, `beta`, `r_est`, `matvecs`, `time`, …) was fixed in week 1 so the
experiment drivers could be written before the solvers were finished.

---

## 12. References

1. **Austin, C., Pollock, S., & Zhu, Y.** (2024). Dynamically accelerating the
   power iteration with momentum. *Numerical Linear Algebra with Applications*,
   31(6), e2584. — the base paper.
2. **Xu, P., He, B., De Sa, C., Mitliagkas, I., & Ré, C.** (2018). Accelerated
   stochastic power iteration. *AISTATS*. — introduces `β = λ₂²/4`.
3. **Rabbani, T., Jain, A., Rajkumar, A., & Huang, F.** (2021). Practical and
   fast momentum-based power methods (DMPower). *MSML*.
4. **Haveliwala, T., & Kamvar, S.** (2003). *The second eigenvalue of the Google
   matrix.* Stanford tech report. — gives `|λ₂(G)| ≤ d`, the bound our
   safeguard and our free `β_opt` both rest on.
5. **Page, L., Brin, S., Motwani, R., & Winograd, T.** (1999). *The PageRank
   citation ranking: bringing order to the web.*
6. **Leskovec, J., & Krevl, A.** (2014). *SNAP Datasets.*
   <https://snap.stanford.edu/data>
7. **Davis, T. A., & Hu, Y.** (2011). The University of Florida sparse matrix
   collection. *ACM TOMS*, 38(1).

---

## 13. Appendix: file inventory and data schemas

### 13.1 Source files

| file | lines | role |
|---|---|---|
| `src/iterations.py` | 698 | Algorithms 1.1/1.2/3.1, safeguards, rate formulas |
| `src/datasets.py` | 241 | SNAP + SuiteSparse acquisition, synthetic fallback |
| `src/google_matrix.py` | 236 | `GoogleMatrix`, `CountingOperator`, edge-list → CSR |
| `src/spectrum.py` | 168 | `μ(λ)`, true rate vs `β`, stability limits, spectrum sampling |
| `src/metrics.py` | 115 | ℓ1/ℓ∞ error, precision@k, Kendall tau, Spearman |
| `src/pagerank_methods.py` | 76 | The five configurations, defined once |
| `src/plotting.py` | 66 | Shared style, Okabe–Ito colour-blind-safe palette |
| `experiments/*.py` | 1,600 | Seven drivers + shared helpers |
| `tests/*.py` | 247 | 19 unit tests |

### 13.2 Result files

| file | one row per | key columns |
|---|---|---|
| `exp1_reproduce.csv` | (matrix, method) | `matvecs`, `r`, `converged`, plus `matvecs_min/max/std` for random starts |
| `exp2_pagerank.csv` | (graph, method) | `matvecs`, `speedup`, `diverged`, `l1_agreement`, `precision_at_100`, `lambda2_measured` |
| `exp3_damping.csv` | (graph, d, method) | `matvecs`, `speedup`, `predicted_speedup`, `observed_rate`, `beta_final` |
| `exp4_ranking.csv` | (d, method) | `matvecs_residual`, `matvecs_precision`, `matvecs_kendall`, `ranking_speedup` |
| `exp4_ranking_curves.csv` | (d, method, step) | per-iterate `residual`, `l1_error`, `precision_at_k`, `kendall` |
| `exp5_spectrum.csv` | (graph, d) | `rate_at_beta_paper`, `diverges`, `beta_best`, `measured_speedup_at_beta_best`, `n_modes_overtaking` |
| `exp6_symmetric.csv` | (problem, method) | `matvecs`, `speedup`, `predicted_speedup` |
| `exp7_normality.csv` | (r, theta, method) | `matvecs`, `speedup`, `symmetric`, `mu_subdominant`, `mu_dominant`, `theory_diverges` |
| `run.log` | — | Full console transcript of the last `./run_all.sh` |

### 13.3 Datasets

| dataset | n | edges | kind | used for |
|---|---|---|---|---|
| `wiki-Vote` | 7,115 | 103,689 | social | Debug graph; the `λ₂ < d` counterexample |
| `cit-HepTh` | 27,770 | 352,807 | citation | Available, unused |
| `cit-HepPh` | 34,546 | 421,578 | citation | **Primary citation case** |
| `web-Stanford` | 281,903 | 2,312,497 | web | **Primary web case** |
| `web-NotreDame` | 325,729 | 1,497,134 | web | Available, unused |
| `web-Google` | 875,713 | 5,105,039 | web | Available, untested |
| `ca-HepPh` | 12,008 | 118,521 | collaboration (undirected) | **Symmetric control** |
| `Kuu`, `Muu`, `ash292`, `bcspwr06` | 292–7,102 | — | SuiteSparse | **Phase 1 reproduction** |
