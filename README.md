# Accelerating PageRank with Dynamic Momentum Power Iteration

CSE 402 Numerical Analysis Sessional — Section C

A methodological extension and cross-domain application of

> Austin, C., Pollock, S., & Zhu, Y. (2024). *Dynamically accelerating the power
> iteration with momentum.* Numerical Linear Algebra with Applications, 31(6),
> e2584. [doi:10.1002/nla.2584](https://doi.org/10.1002/nla.2584)

**Group members:** Nakib Arman (2105128), Ali Asif Khan (2105131),
Shariar Al Kabir (2105132), Dip Saha (2105138), MD. Mehedi Hasan Mim (2105142).
**Presenter:** Dip Saha.

---

## What this project asked

The base paper proposes a dynamic momentum method (its Algorithm 3.1) that
accelerates the power iteration for the dominant eigenpair. The momentum
parameter `β_k` is re-estimated every iteration from the Rayleigh quotient and
the ratio of the last two residuals, so no prior knowledge of `λ₂` is needed,
and each iteration still costs exactly one matrix-vector product. The paper
benchmarks it only on abstract SuiteSparse matrices.

PageRank is the obvious real application: it *is* a dominant-eigenvector problem
on a huge sparse matrix, and it is slowest precisely when the two largest
eigenvalues are close — the regime the method is built for. This project
implements the method and applies it to real citation and web graphs.

## What we found

The headline result is **negative, and the reason is specific and
demonstrable**: the dynamic momentum method does not transfer to PageRank, and
the obstruction is that the Google matrix is non-normal.

1. **The method works exactly as advertised on the problems the paper covers.**
   On `diag(1000:-1:1)` we measure a **40.4×** speedup over the power iteration
   against a predicted 44.7×, with the dynamic method (683 matvecs) beating even
   the optimal static parameter (777) — reproducing the paper's central claim.

2. **For the Google matrix the optimal static parameter is exactly `d²/4`, for
   free.** `λ₁ = 1` by stochasticity, and Haveliwala & Kamvar (2003) give
   `|λ₂| ≤ d`. We *measured* `|λ₂(G)| = d` to six decimal places on `cit-HepPh`
   and `web-Stanford`, so no eigensolve is needed. (On the social graph
   `wiki-Vote` the bound is slack: `λ₂/d = 0.59`.)

3. **Algorithm 3.1 as printed diverges on every real graph we tried.** The
   residual oscillates, a single step with `d_{k+1} > d_k` pins `ρ_k` at 1,
   hence `r_{k+1} = 1` and `β_k → λ₁²/4 = 1/4` — the exact value at which
   section 2 of the paper shows convergence stops. On `wiki-Vote` the residual
   bottoms out near `10⁻³` at iteration 20 and climbs back to `1.5 × 10⁻¹`.

4. **The deeper cause is complex eigenvalues.** The momentum iteration is a
   power iteration on the augmented matrix, with eigenvalues
   `μ± = (λ ± √(λ² − 4β))/2`. For a *real* spectrum every `λ` with `λ²/4 < β`
   maps to modulus exactly `√β`, which is what makes `β = λ₂²/4` optimal. For
   complex `λ` that identity fails upward: at `λ = iy` the modulus is
   `(|y| + √(y² + 4β))/2 > √β`, growing without bound in `|y|`.

   So the destabilising modes are **not** near the top of the spectrum but are
   small-modulus eigenvalues near the imaginary axis — which a
   largest-magnitude eigensolver never even returns. On `wiki-Vote` at
   `d = 0.99`, eigenvalues of modulus **0.23** reach `|μ| = 0.624` against
   `|μ_λ₁| = 0.571`, so the iteration converges to the wrong eigenmode.

5. **The control confirms this is the problem, not the code.** On the *same*
   `cit-HepPh` data symmetrised as the co-citation matrix `AAᵀ`, the dynamic
   method delivers **3.58×** against a predicted 3.79×. The only thing that
   changed is normality.

6. **Our safeguarded variant restores robustness**, using only information
   PageRank supplies for free (`λ₁ = 1`, `r ≤ d`) plus residual-triggered
   backtracking. It converges on every graph and damping factor tested, reaching
   4.08× on `cit-HepPh` at `d = 0.99` where the published algorithm diverges —
   though on other graphs it is merely competitive with plain power iteration.

7. **For ranking, none of this matters much.** The top-100 ordering is already
   correct after **7–16** matrix-vector products, while driving the
   eigen-residual to `10⁻¹²` takes up to **2005**. Ranking converges 15–286×
   sooner than the residual, so the acceleration the paper offers is spent on
   digits nobody reads.

## Headline numbers

Reproduction of the base paper (`exp1`), matrix-vector products to residual
`10⁻¹²`, started from `u₀ = (1,…,1)ᵀ`:

| matrix | r = \|λ₂/λ₁\| | power | static β=λ₂²/4 | **dynamic (Alg. 3.1)** |
|---|---|---|---|---|
| `diag(1000:-1:1)` | 0.9990 | 27619 | 777 | **683** |
| `Kuu` | 0.9981 | >2000 | 1166 | **528** |
| `Muu` | 0.9992 | >2000 | 728 | **292** |
| `ash292` | 0.9153 | 313 | 75 | **73** |
| `bcspwr06` | 0.9814 | 1360 | 154 | **145** |

Our measured spectral ratios match every value the paper reports, and the
dynamic method beats the optimal static parameter on all five — the paper's
central claim, reproduced. The paper's Table 1 finding that the dynamic method
is *more* sensitive to the initial vector on indefinite matrices reproduces too:
over 30 random starts on `diag(linspace(-99,100,200))` its matvec count has
standard deviation 118 against the static method's 9.

All eigensolves use a fixed ARPACK start vector, so every number here is
reproducible run to run rather than drifting with `λ₂`'s accuracy.

PageRank at `d = 0.85` (`exp2`), same units:

| graph | n | \|λ₂(G)\| | power | static β=d²/4 | Alg. 3.1 | **safeguarded** |
|---|---|---|---|---|---|---|
| `wiki-Vote` | 7,115 | 0.5014 | **39** | 78 | diverged | 68 |
| `cit-HepPh` | 34,546 | 0.8500 | **126** | 162 | diverged | 153 |
| `web-Stanford` | 281,903 | 0.8500 | **152** | diverged | diverged | 312 |

Nothing beats the plain power iteration. Across the damping sweep (`exp3`) the
safeguarded variant reaches 2.28× at `d = 0.95` and 4.08× at `d = 0.99` on
`cit-HepPh`, against 6.30× and 14.13× predicted by the real-spectrum theory,
and is below parity elsewhere.

The symmetric control (`exp6`) is the contrast that makes the cause clear:

| problem | r | power | static | **dynamic** | predicted |
|---|---|---|---|---|---|
| `ca-HepPh` eigenvector centrality | 0.3751 | 35 | 23 | **23** (1.52×) | 1.67× |
| `cit-HepPh` co-citation `AAᵀ` | 0.8638 | 236 | 72 | **66** (3.58×) | 3.79× |

Same citation data, symmetrised: the acceleration appears.

## Repository layout

```
src/
  iterations.py        Algorithms 1.1, 1.2, 3.1 + safeguards (matrix-agnostic)
  google_matrix.py     Implicit Google matrix operator, matvec counting
  pagerank_methods.py  The five configurations compared everywhere
  spectrum.py          Augmented-matrix eigenvalues, true rate vs beta
  datasets.py          SNAP / SuiteSparse download + cache, synthetic fallback
  metrics.py           l1, Kendall tau, precision@k
  plotting.py          Shared figure style
experiments/
  exp1_reproduce.py    Phase 1 gate: reproduce the paper
  exp2_pagerank.py     Phase 2 gate: validate G; Phase 3: benchmark at d=0.85
  exp3_damping.py      Phase 4: damping sweep, measured vs predicted speedup
  exp4_ranking.py      Phase 5: residual convergence vs ranking convergence
  exp5_spectrum.py     Phase 6: why beta = d^2/4 fails, with empirical check
  exp6_symmetric.py    Phase 7: symmetric control
tests/                 19 unit tests, including regressions for both safeguard bugs
results/               CSV output
figures/               PDF (for LaTeX) and PNG (for slides)
```

`src/iterations.py` never imports anything about PageRank; it takes a linear
operator. That is what lets the SuiteSparse reproduction and the Google-matrix
study share one code path, and what makes the symmetric control free.

## Running it

```bash
make setup     # .venv + pinned dependencies
make test      # 19 unit tests, ~1 s
make all       # every experiment; downloads ~90 MB of SNAP data on first run
make quick     # offline subset on synthetic graphs, no network needed
```

Individual phases: `make phase1` … `make phase6`. Every script takes
`--graphs`, and `--offline` to substitute synthetic graphs.

Experiments 1 and 2 print an explicit **PASS/FAIL gate**. Phase 1 checks the
reproduction against the paper; Phase 2 checks the Google matrix against
`networkx.pagerank` *and* against its own fixed-point residual
`‖Gx − x‖₁ < 10⁻¹²`. If either fails, nothing downstream is meaningful.

## Two implementation traps worth knowing

**The matvec shortcut.** Standard PageRank code writes
`y = d·Px + (d·aᵀx + (1−d))/n`, valid only because plain power iteration
preserves `‖x‖₁ = 1`. Momentum iterates are 2-normalised and may go negative, so
the honest operator
`y = d·Px + (d/n)(aᵀx)e + ((1−d)/n)(eᵀx)e` must be used. `test_google_matrix.py`
asserts `eᵀGx = eᵀx` for arbitrary `x`, including negative and zero vectors.

**Capping `β_k` against `ν_k`.** Bounding `β_k ≤ θ·ν_k²/4` looks like a
harmless way to enforce `β < λ₁²/4`. It is not: since `ν_k → λ₁`, any `θ < 1`
clips `β_k` below the optimum whenever `r² > θ` — exactly the near-degenerate
regime the method exists for. On `diag(1000:-1:1)` a 1% clip degrades the
asymptotic rate from 0.9562 to 0.9895 and quadruples the iteration count. The
cap must be taken against a *known* `λ₁`, which PageRank provides.

## Data

SNAP graphs are downloaded on demand to `data/` and cached: `wiki-Vote` (7k),
`cit-HepTh` (28k), `cit-HepPh` (35k), `web-Stanford` (282k), `web-NotreDame`
(326k), `web-Google` (876k), `ca-HepPh` (12k, undirected). SuiteSparse matrices
from the paper's own test suites come via `ssgetpy`. If the network is
unavailable every script falls back to a synthetic generator with power-law
out-degrees and dangling nodes.
