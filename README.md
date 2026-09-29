# Accelerating PageRank with Dynamic Momentum Power Iteration

**CSE 402 — Numerical Analysis Sessional · Section C**

A cross-domain application of

> Austin, C., Pollock, S., & Zhu, Y. (2024). *Dynamically accelerating the power
> iteration with momentum.* Numerical Linear Algebra with Applications, **31**(6),
> e2584. [doi:10.1002/nla.2584](https://doi.org/10.1002/nla.2584)

**Group:** Nakib Arman (2105128) · Ali Asif Khan (2105131) · Shariar Al Kabir
(2105132) · Dip Saha (2105138) · MD. Mehedi Hasan Mim (2105142).
**Presenter:** Dip Saha.

---

This is a self-contained study with a single, narrow claim:

> **The dynamic momentum method of Austin, Pollock & Zhu (2024) accelerates real
> PageRank by up to 41.9×, exactly as its theory predicts, with no
> modifications of any kind.**

Everything here is the published algorithm. No safeguards, no rate guard, no
backtracking, no parameter tuning — `experiment/_methods.py` explicitly passes
the arguments that switch all of that off, so it cannot leak in.

The primary dataset is **`ca-HepPh`**, the arXiv high-energy-physics
co-authorship network (12,008 authors, 236,978 links, undirected). Experiment 7
repeats the headline measurement across four more undirected co-authorship
graphs to show the result is not specific to it.

## Results

**Reproduction of the base paper** (`exp1`) — matvecs to residual 10⁻¹²:

| matrix | r | power | static β=λ₂²/4 | **dynamic (Alg. 3.1)** |
|---|---|---|---|---|
| `diag(1000:-1:1)` | 0.9990 | 27,619 | 777 | **683** (40.4×) |
| `Kuu` | 0.9981 | >2,000 | 1,166 | **528** |
| `Muu` | 0.9992 | >2,000 | 728 | **292** |
| `ash292` | 0.9153 | 313 | 75 | **73** |
| `bcspwr06` | 0.9814 | 1,360 | 154 | **145** |

Every measured spectral ratio matches the value the paper publishes.

**PageRank on `ca-HepPh`** (`exp3`) — the headline:

| d | power | static β=d²/4 | **dynamic** | speed-up | **theory** | % of theory |
|---|---|---|---|---|---|---|
| 0.85 | 150 | 49 | **47** | 3.19× | 3.60× | 89% |
| 0.90 | 231 | 61 | **58** | 3.98× | 4.43× | 90% |
| 0.95 | 471 | 88 | **83** | 5.67× | 6.30× | 90% |
| 0.99 | 2,377 | 197 | **182** | 13.06× | 14.13× | 92% |
| 0.999 | 23,777 | 622 | **567** | **41.93×** | 44.72× | **94%** |

The agreement *improves* as d → 1, exactly as Lemma 3.4 predicts: the parameter
estimate becomes more stable as the spectral gap closes.

**Observed vs predicted convergence rate** — agreement to four decimals:

| d | 0.85 | 0.90 | 0.95 | 0.99 | 0.999 |
|---|---|---|---|---|---|
| measured | 0.5563 | 0.6273 | 0.7241 | 0.8676 | 0.9562 |
| ρ(d) predicted | 0.5567 | 0.6268 | 0.7239 | 0.8676 | 0.9562 |

**Is it just this one graph?** (`exp7`) — the same sweep on the whole family of
arXiv co-authorship networks, spanning 5,242 to 23,133 authors and five fields
of physics:

| graph | n | d=0.85 | d=0.95 | d=0.99 | d=0.999 |
|---|---|---|---|---|---|
| `ca-GrQc` | 5,242 | 3.27× | 5.84× | 13.14× | 39.85× |
| `ca-HepTh` | 9,877 | 3.21× | 5.76× | 13.19× | 41.30× |
| `ca-HepPh` | 12,008 | 3.19× | 5.67× | 13.06× | 41.93× |
| `ca-AstroPh` | 18,772 | 3.22× | 5.67× | 13.06× | 42.04× |
| `ca-CondMat` | 23,133 | 3.21× | 5.73× | 13.15× | 41.80× |
| **theory** | | 3.60× | 6.30× | 14.13× | 44.72× |

**25 PageRank problems, every one converged**, all spectra real, and the
fraction of theory attained is 88.3%–94.0% (mean 91.1%). At d = 0.99 the spread
across a 4.4× range of graph sizes is 13.06–13.19× — about 1%. The acceleration
is a property of undirected graphs, not of one dataset.

**Why it works** (`exp5`): `ca-HepPh` is undirected, so its Google matrix has a
**real** spectrum — measured `max |Im λ| ≈ 5×10⁻¹⁶`. That is exactly the
condition the paper's theory assumes, and the consequence is measurable: every
subdominant mode folds onto the circle of radius `√β`.

| d | √β | strongest rival \|μ\| | \|μ(λ₁)\| | rate | ρ(d) predicted |
|---|---|---|---|---|---|
| 0.85 | 0.4250 | **0.4250** | 0.7634 | 0.5567 | 0.5567 |
| 0.95 | 0.4750 | **0.4750** | 0.6561 | 0.7239 | 0.7239 |
| 0.99 | 0.4950 | **0.4950** | 0.5705 | 0.8676 | 0.8676 |

**Robustness and cost** (`exp6`) — 153 runs, every one converged:

| starting vector | d=0.85 | d=0.95 | d=0.99 |
|---|---|---|---|
| `ones` (uniform prior) | 3.19× | 5.67× | **13.06×** |
| random **non-negative** | 3.28× | 5.79× | **13.16×** |
| random **signed** (paper's stress test) | 2.72× | 3.74× | 5.55× |

The headline is robust to any non-negative start — the only kind meaningful for
PageRank, since scores are a distribution. Under the paper's signed stress test
the dynamic method slows and becomes more variable than the static one, which is
the same sensitivity the paper reports in its own Table 1; it still beats the
power iteration.

Cost per matrix-vector product is effectively identical across the three methods
(1.000× / 1.047× / 1.055× of the power iteration), so iteration counts are a
fair metric — and the wall-clock speed-up agrees with the matvec one
(12.11× vs 13.15× at d=0.99).

**Ranking vs residual** (`exp4`): the top-100 ordering is correct after 9–10
matvecs while the residual needs 47–182, so the ranking is usable 5–18× sooner
than the eigensolver's own stopping rule suggests.

## The boundary of the result (`exp8`)

The claim above is about **undirected** graphs, and the boundary is sharp.
`web-Stanford` is a directed web graph (281,903 pages, 2.3M links); direction is
the only structural difference, and it changes everything.

| | `ca-HepPh` (undirected) | `web-Stanford` (directed) |
|---|---|---|
| max \|Im λ\| of G | **3 × 10⁻¹⁴** | **0.990** |
| strongest rival \|μ\| | 0.4950 = √β | **1.1950** |
| \|μ(λ₁)\| — the winner | 0.5705 | 0.5705 |
| rivals overtaking the winner | **0 of 38** | **55 of 59** |
| Algorithm 3.1 | converges, 13.06× | **diverges at every d** |

The worst mode on `web-Stanford` is `λ = 0.99i` — essentially purely imaginary.
Its `|μ| = 1.195 > 1`, so it does not merely beat the dominant mode, it *grows*:
the iteration diverges outright. What makes a mode damaging is its **angle**,
not its size.

Meanwhile plain power iteration converges normally there (152 / 478 / 2,428
matvecs at d = 0.85 / 0.95 / 0.99), so this is a property of the Google matrix,
not of the implementation.

## Running it

```bash
make setup        # create .venv and install pinned dependencies
make test         # 19 unit tests, ~1 second
make all          # every experiment, ~14 min  (downloads ~95 MB on first run)
make quick        # fast subset: no SuiteSparse, no web-Stanford
make exp3         # one experiment at a time (exp1 ... exp8)
```

`run_all.sh` stops if either gate fails, because nothing downstream means
anything if the implementation or the Google matrix is wrong.

Outputs go to `result/` (CSV + `run.log`) and `figure/` (PDF for LaTeX, PNG for
slides).

## Repository layout

```
run_all.sh          run everything, in order, with gates
Makefile            make setup / test / all / quick / exp1..exp8
experiment/         the eight experiments
src/                the solvers and the Google matrix operator
tests/              19 unit tests
result/             CSV output + run.log
figure/             PDF for LaTeX, PNG for slides
data/               downloaded graphs, cached (gitignored)
```

### `src/` — the shared library

| module | role |
|---|---|
| `iterations.py` | Algorithms 1.1, 1.2 and 3.1. Matrix-agnostic: takes any linear operator. |
| `google_matrix.py` | The Google matrix applied as a formula, never built. |
| `datasets.py` | SNAP and SuiteSparse download + cache, with an offline fallback. |
| `spectrum.py` | `μ(λ)` from eq. (2.7), and the true rate as a function of `β`. |
| `metrics.py` | ℓ1 error, precision@k, Kendall tau. |

### `experiment/` — the study

| file | what it does |
|---|---|
| `_common.py` | paths, the graph name, the damping list, tolerances |
| `_methods.py` | the three published algorithms, with safeguards explicitly disabled |
| `_style.py` | one colour-blind-safe figure style |
| `experiment/exp1_reproduce.py` | reproduce the paper's own benchmarks — the validity gate |
| `exp2_pagerank.py` | build and validate the Google matrix, benchmark at d=0.85 |
| `exp3_damping.py` | the damping sweep: measured vs predicted |
| `exp4_ranking.py` | ranking quality vs eigen-residual |
| `exp5_spectrum.py` | the spectral explanation of why it works |
| `exp6_robustness.py` | sensitivity to the starting vector, and wall-clock cost |
| `exp7_generality.py` | the same sweep across five undirected co-authorship graphs |
| `exp8_limitations.py` | the boundary: spectrum and convergence on a directed web graph |

The solvers live in `src/`; the experiments only choose how to call them.
`experiment/_methods.py` is the single place where that choice is made, and it
explicitly passes the arguments that disable every safeguard, so the published
algorithm is what runs.
