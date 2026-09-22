"""Construction of the Google matrix as an implicit linear operator.

The Google matrix of a directed graph with ``n`` nodes and damping factor ``d``
is, in the column-stochastic convention used throughout this project,

    G = d * (P + (1/n) e a^T) + ((1 - d)/n) e e^T                        (1)

where ``P[i, j] = 1 / outdeg(j)`` when ``j -> i``, ``a_j = 1`` exactly when node
``j`` is dangling (no out-links), and ``e`` is the all-ones vector.  The middle
term patches dangling columns into uniform columns so that ``G`` is stochastic;
the last term is the teleportation / personalisation term.

``G`` is dense, so it is never formed.  :class:`GoogleMatrix` applies (1) as

    y = d * (P @ x) + (d / n) * (a . x) * e + ((1 - d) / n) * (e . x) * e

at a cost of one sparse matrix-vector product plus two dot products.

.. warning::

   Standard PageRank implementations shortcut (1) to
   ``y = d * (P @ x) + (d * (a . x) + (1 - d)) / n`` by assuming ``||x||_1 = 1``,
   which plain power iteration preserves.  **The momentum methods break that
   invariant**: their iterates are 2-normalised and may contain negative
   entries, so ``e . x`` is neither one nor even necessarily positive.  Using
   the shortcut silently produces a wrong operator and therefore a wrong
   ranking.  This module always evaluates the honest form above.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

__all__ = [
    "GoogleMatrix",
    "CountingOperator",
    "build_google_matrix",
    "edges_to_sparse",
    "normalise_ranking",
]


# --------------------------------------------------------------------------
# Generic matvec-counting wrapper (used for the SuiteSparse experiments)
# --------------------------------------------------------------------------


class CountingOperator:
    """Wrap any linear operator and count the matrix-vector products.

    The comparison in the paper is fundamentally about *matrix-vector products*,
    not iterations, because the competing DMPower method spends three of them
    per iteration.  Counting them in one place removes any chance of the
    bookkeeping drifting between methods.
    """

    def __init__(self, A, name: str = ""):
        self._A = A
        self.shape = A.shape
        self.dtype = getattr(A, "dtype", np.dtype(float))
        self.name = name
        self.count = 0

    def reset(self) -> None:
        self.count = 0

    def matvec(self, x: np.ndarray) -> np.ndarray:
        self.count += 1
        return self._A @ x

    def __matmul__(self, x: np.ndarray) -> np.ndarray:
        return self.matvec(x)

    def __repr__(self) -> str:
        return f"CountingOperator({self.name or type(self._A).__name__}, n={self.shape[0]}, count={self.count})"


# --------------------------------------------------------------------------
# Google matrix
# --------------------------------------------------------------------------


@dataclass
class GoogleMatrix:
    """Implicit Google matrix ``G`` of equation (1).

    Attributes
    ----------
    P:
        Column-stochastic link matrix in CSR format, with dangling columns left
        entirely zero (they are handled analytically by ``dangling``).
    dangling:
        Boolean mask of the dangling columns.
    d:
        Damping factor.
    n:
        Number of nodes.
    count:
        Matrix-vector products applied so far.
    """

    P: sp.csr_matrix
    dangling: np.ndarray
    d: float
    n: int
    name: str = ""
    count: int = 0

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n, self.n)

    @property
    def dtype(self):
        return self.P.dtype

    def reset(self) -> None:
        self.count = 0

    def with_damping(self, d: float) -> "GoogleMatrix":
        """Return the same graph at a different damping factor.

        Re-using ``P`` across the damping sweep avoids rebuilding the sparse
        structure five times per graph, which dominates the runtime on
        ``web-Google``.
        """
        return GoogleMatrix(
            P=self.P, dangling=self.dangling, d=float(d), n=self.n, name=self.name
        )

    def matvec(self, x: np.ndarray) -> np.ndarray:
        """Apply ``G`` without assuming anything about the scaling of ``x``."""
        self.count += 1
        # Preserve the input dtype: ARPACK probes this operator with complex
        # vectors when asked for complex eigenpairs, and silently discarding the
        # imaginary part would corrupt the spectral diagnostics in src.spectrum.
        x = np.asarray(x).ravel()
        if not np.issubdtype(x.dtype, np.number):
            x = x.astype(float)
        y = self.d * (self.P @ x)
        # (d / n) * (a . x)  from the dangling-column patch
        # ((1 - d) / n) * (e . x)  from teleportation
        dangling_mass = x[self.dangling].sum() if self.dangling.any() else 0.0
        total_mass = x.sum()
        y += (self.d * dangling_mass + (1.0 - self.d) * total_mass) / self.n
        return y

    def __matmul__(self, x: np.ndarray) -> np.ndarray:
        return self.matvec(x)

    def __repr__(self) -> str:
        return (
            f"GoogleMatrix({self.name!r}, n={self.n}, nnz={self.P.nnz}, "
            f"d={self.d}, dangling={int(self.dangling.sum())})"
        )


def edges_to_sparse(
    src: np.ndarray, dst: np.ndarray, n: int | None = None
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Turn an edge list into a column-stochastic link matrix.

    SNAP node identifiers are arbitrary integers with gaps, so they are first
    remapped onto ``0..n-1``.

    Returns
    -------
    P:
        Column-stochastic CSR matrix; dangling columns are all-zero.
    dangling:
        Boolean mask of dangling columns.
    node_ids:
        The original identifiers, indexed by the new contiguous index, so that
        a ranking can be reported in terms of the source data.
    """
    src = np.asarray(src, dtype=np.int64).ravel()
    dst = np.asarray(dst, dtype=np.int64).ravel()

    node_ids, remapped = np.unique(np.concatenate([src, dst]), return_inverse=True)
    n_found = node_ids.size
    half = src.size
    src_i = remapped[:half]
    dst_i = remapped[half:]
    if n is None:
        n = n_found

    # Drop self-loops and duplicate edges: both are artefacts of the raw dumps
    # and neither belongs in a link matrix.
    keep = src_i != dst_i
    src_i, dst_i = src_i[keep], dst_i[keep]
    pairs = np.unique(np.stack([src_i, dst_i]), axis=1)
    src_i, dst_i = pairs[0], pairs[1]

    outdeg = np.bincount(src_i, minlength=n).astype(float)
    dangling = outdeg == 0
    # P[dst, src] = 1 / outdeg(src): columns are sources, rows are targets.
    weights = 1.0 / outdeg[src_i]
    P = sp.csr_matrix((weights, (dst_i, src_i)), shape=(n, n))
    P.sum_duplicates()
    return P, dangling, node_ids


def build_google_matrix(
    src: np.ndarray,
    dst: np.ndarray,
    d: float = 0.85,
    n: int | None = None,
    name: str = "",
) -> tuple[GoogleMatrix, np.ndarray]:
    """Build a :class:`GoogleMatrix` from a raw edge list.

    Returns the operator together with the original node identifiers.
    """
    P, dangling, node_ids = edges_to_sparse(src, dst, n=n)
    G = GoogleMatrix(P=P, dangling=dangling, d=float(d), n=P.shape[0], name=name)
    return G, node_ids


def normalise_ranking(x: np.ndarray) -> np.ndarray:
    """Convert a 2-normalised eigenvector into a PageRank probability vector.

    The dominant eigenvector of ``G`` is positive (Perron-Frobenius, ``d < 1``),
    but an eigensolver -- and in particular a momentum iteration -- may return
    it scaled by ``-1``.  Fix the sign from the total mass, then rescale to unit
    1-norm.
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.sum() < 0:
        x = -x
    total = np.abs(x).sum()
    if total == 0:
        raise ValueError("cannot normalise the zero vector")
    return x / total
