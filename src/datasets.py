"""Dataset acquisition and caching.

Two families of problems are used:

* **SuiteSparse matrices** -- the benchmarks the base paper itself reports on.
  These drive the Phase 1 reproduction gate: if our Algorithm 3.1 does not
  behave on ``Kuu``/``ash292``/``bcspwr06`` the way Figure 3 and Table 1 of the
  paper say it should, the implementation is wrong and nothing downstream is
  worth running.

* **SNAP graphs** -- real citation and web networks, the cross-domain
  application that the base paper never attempts.

Every download is cached under ``data/``.  If the network is unavailable, the
synthetic generators at the bottom of this module produce graphs with the same
structural features that matter here (power-law out-degrees, dangling nodes,
multiple closed subsets) so the whole pipeline remains runnable offline.
"""

from __future__ import annotations

import gzip
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "SNAP_DATASETS",
    "PAPER_MATRICES",
    "DATA_DIR",
    "load_snap_edges",
    "load_suitesparse",
    "synthetic_web_graph",
    "diag_benchmark",
]

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass(frozen=True)
class SnapDataset:
    key: str
    url: str
    nodes: int
    edges: int
    kind: str        # "citation", "web", "social" or "collaboration"
    directed: bool
    note: str


# Ordered from smallest to largest.  Work up this list: `wiki-Vote` is the
# debugging graph, `cit-HepPh` is the citation case named in the proposal,
# and `web-Google` is the ceiling for a laptop.
SNAP_DATASETS: dict[str, SnapDataset] = {
    "wiki-Vote": SnapDataset(
        "wiki-Vote", "https://snap.stanford.edu/data/wiki-Vote.txt.gz",
        7_115, 103_689, "social", True,
        "Wikipedia adminship votes; small enough to validate against NetworkX.",
    ),
    "cit-HepTh": SnapDataset(
        "cit-HepTh", "https://snap.stanford.edu/data/cit-HepTh.txt.gz",
        27_770, 352_807, "citation", True,
        "arXiv hep-th citation network.",
    ),
    "cit-HepPh": SnapDataset(
        "cit-HepPh", "https://snap.stanford.edu/data/cit-HepPh.txt.gz",
        34_546, 421_578, "citation", True,
        "arXiv hep-ph citation network; the proposal's primary citation case.",
    ),
    "web-Stanford": SnapDataset(
        "web-Stanford", "https://snap.stanford.edu/data/web-Stanford.txt.gz",
        281_903, 2_312_497, "web", True,
        "stanford.edu web graph.",
    ),
    "web-NotreDame": SnapDataset(
        "web-NotreDame", "https://snap.stanford.edu/data/web-NotreDame.txt.gz",
        325_729, 1_497_134, "web", True,
        "nd.edu web graph; unusually many dangling nodes.",
    ),
    "web-Google": SnapDataset(
        "web-Google", "https://snap.stanford.edu/data/web-Google.txt.gz",
        875_713, 5_105_039, "web", True,
        "Google programming-contest web graph; the largest case we run.",
    ),
    "ca-HepPh": SnapDataset(
        "ca-HepPh", "https://snap.stanford.edu/data/ca-HepPh.txt.gz",
        12_008, 118_521, "collaboration", False,
        "Undirected co-authorship graph; the symmetric control problem, where "
        "the paper's acceleration theorems actually apply.",
    ),
}


# The SuiteSparse matrices reported in the paper, by test suite.
PAPER_MATRICES = {
    "Kuu": dict(group="MathWorks", n=7102, r=0.9981, suite=1),
    "Muu": dict(group="MathWorks", n=7102, r=0.9992, suite=1),
    "ash292": dict(group="HB", n=292, r=0.9153, suite=2),
    "bcspwr06": dict(group="HB", n=1454, r=0.9814, suite=2),
}


# --------------------------------------------------------------------------
# Download helpers
# --------------------------------------------------------------------------


def _cache_path(key: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / f"{key}.txt.gz"


def _download(url: str, dest: Path, timeout: float = 60.0) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "cse402-pagerank/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as fh:
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
    os.replace(tmp, dest)


def load_snap_edges(key: str, allow_download: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(src, dst)`` arrays for a SNAP dataset, downloading if needed.

    The files are whitespace-separated integer pairs with ``#`` comment headers.
    They are parsed with :func:`numpy.loadtxt` rather than NetworkX, which is
    orders of magnitude too slow above roughly 100k nodes.
    """
    if key not in SNAP_DATASETS:
        raise KeyError(f"unknown dataset {key!r}; choose from {sorted(SNAP_DATASETS)}")
    ds = SNAP_DATASETS[key]
    path = _cache_path(key)

    if not path.exists():
        if not allow_download:
            raise FileNotFoundError(f"{path} missing and downloads are disabled")
        print(f"  downloading {ds.url} ...", flush=True)
        _download(ds.url, path)

    with gzip.open(path, "rt") as fh:
        data = np.loadtxt(fh, dtype=np.int64, comments="#")

    src, dst = data[:, 0], data[:, 1]
    if not ds.directed:
        # Store both orientations so the resulting link matrix is symmetric.
        src, dst = np.concatenate([src, dst]), np.concatenate([dst, src])
    return src, dst


def load_suitesparse(name: str, quiet: bool = True):
    """Fetch a SuiteSparse matrix by name and return it as a SciPy sparse matrix.

    ``ssgetpy`` draws a tqdm progress bar on stderr, which emits several hundred
    lines per matrix and drowns the experiment output when a run is logged to a
    file.  It is silenced by default.
    """
    import contextlib
    import io
    import os

    import ssgetpy
    from scipy.io import mmread

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    results = ssgetpy.search(name=name, limit=10)
    exact = [m for m in results if m.name == name]
    if not exact:
        raise LookupError(f"SuiteSparse matrix {name!r} not found")
    matrix = exact[0]

    sink = io.StringIO() if quiet else None
    with (contextlib.redirect_stderr(sink) if quiet else contextlib.nullcontext()):
        matrix.download(format="MM", destpath=str(DATA_DIR), extract=True)

    # ssgetpy reports the tarball rather than the directory it extracted into,
    # so locate the Matrix Market file ourselves.  The archive unpacks to
    # ``data/<name>/<name>.mtx``; the recursive glob also covers collections
    # that nest the file one level deeper.
    candidates = sorted(DATA_DIR.glob(f"{name}/**/*.mtx"))
    exact_name = [f for f in candidates if f.stem == name]
    chosen = (exact_name or candidates)
    if not chosen:
        raise FileNotFoundError(f"no .mtx file for {name!r} under {DATA_DIR}")
    return mmread(str(chosen[0])).tocsr()


# --------------------------------------------------------------------------
# Synthetic problems
# --------------------------------------------------------------------------


def diag_benchmark(n: int = 1000):
    """``A = diag(n : -1 : 1)``, Matrix 1 of the paper's test suite 1.

    Its spectral ratio is ``r = (n-1)/n``, i.e. 0.999 at the default size, and
    every eigenvalue is known exactly -- which makes it the right problem for
    the reproduction gate.
    """
    import scipy.sparse as sp

    values = np.arange(n, 0, -1, dtype=float)
    return sp.diags(values).tocsr(), values


def synthetic_web_graph(
    n: int = 20_000,
    avg_outdeg: float = 8.0,
    dangling_frac: float = 0.1,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a directed graph resembling a real web crawl.

    Out-degrees are power-law distributed, link targets follow preferential
    attachment, and a configurable fraction of nodes are left dangling.  This is
    the offline fallback: it exercises exactly the code paths (dangling-column
    patching, highly skewed degrees) that a real web graph does.
    """
    rng = np.random.default_rng(seed)

    n_dangling = int(round(dangling_frac * n))
    sources = rng.permutation(n)[n - n_dangling:] if n_dangling else np.array([], dtype=int)
    is_dangling = np.zeros(n, dtype=bool)
    is_dangling[sources] = True
    active = np.flatnonzero(~is_dangling)

    # Power-law out-degrees with mean approximately `avg_outdeg`.
    raw = rng.pareto(1.6, size=active.size) + 1.0
    outdeg = np.maximum(1, np.round(raw * avg_outdeg / raw.mean()).astype(int))
    outdeg = np.minimum(outdeg, 200)

    # Preferential-attachment target distribution.
    popularity = rng.pareto(1.3, size=n) + 1.0
    probs = popularity / popularity.sum()

    src = np.repeat(active, outdeg)
    dst = rng.choice(n, size=src.size, p=probs)
    return src, dst
