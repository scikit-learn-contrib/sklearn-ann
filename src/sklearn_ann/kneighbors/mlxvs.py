"""kNN graphs on the GPU of Apple silicon Macs with mlxvs."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, cast

import mlxvs
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import Tags, TargetTags, TransformerTags
from sklearn.utils.validation import validate_data

from ..utils import TransformerChecksMixin, postprocess_knn_csr

if TYPE_CHECKING:
    from typing import Literal, Self

    from numpy.typing import ArrayLike, NDArray


class MetricInfo(NamedTuple):
    metric: str
    sqrt: bool = False
    negate: bool = False


METRIC_MAP: dict[str, MetricInfo] = {
    "euclidean": MetricInfo("l2", sqrt=True),
    "l2": MetricInfo("l2", sqrt=True),
    "sqeuclidean": MetricInfo("l2"),
    "cosine": MetricInfo("cosine", negate=True),
}


class MLXVSTransformer(TransformerChecksMixin, TransformerMixin, BaseEstimator):
    """kNN graphs on the GPU of Apple silicon Macs, with `mlxvs`_.

    Small inputs, and every input with ``algorithm="brute"``, get the exact
    neighbors (a matrix multiply and a top-k on the GPU). Larger ones with
    ``algorithm="auto"`` (or ``"graph"``) get approximate neighbors:
    ``fit_transform`` builds the kNN graph of the rows with NN-Descent in
    overlapping k-means clusters, and ``transform`` searches a CAGRA graph
    index of the fitted rows. Needs macOS 14+ on Apple silicon.

    .. _`mlxvs`: https://github.com/mikepapadim/mlxvs

    Parameters
    ----------
    n_neighbors
        Neighbors per row, not counting the row itself in ``fit_transform``.
    metric
        ``"euclidean"`` (or ``"l2"``), ``"sqeuclidean"`` or ``"cosine"``
        (distance ``1 - cosine similarity``).
    algorithm
        ``"auto"``: exact below ``exact_below`` rows, approximate above.
        ``"brute"``: always exact. ``"graph"``: always approximate.
    exact_below
        With ``algorithm="auto"``, inputs of fewer rows are searched exactly.
    itopk_size
        Length of the graph search's internal list: higher is more accurate and
        slower.
    graph_degree
        Degree of the search graph ``transform`` searches.
    include_fwd, include_rev
        Which directions of the kNN graph to return, as the other transformers.
    """

    # the GPU index transform searches, built on first use
    _index: mlxvs.CagraIndex | None = None

    def __init__(
        self,
        n_neighbors: int = 5,
        *,
        metric: str = "euclidean",
        algorithm: Literal["auto", "brute", "graph"] = "auto",
        exact_below: int = 50_000,
        itopk_size: int = 64,
        graph_degree: int = 32,
        include_fwd: bool = True,
        include_rev: bool = False,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.algorithm = algorithm
        self.exact_below = exact_below
        self.itopk_size = itopk_size
        self.graph_degree = graph_degree
        self.include_fwd = include_fwd
        self.include_rev = include_rev

    @property
    def _metric_info(self) -> MetricInfo:
        return METRIC_MAP[self.metric]

    def _exact(self) -> bool:
        return self.algorithm == "brute" or (
            self.algorithm == "auto" and self.n_samples_fit_ < self.exact_below
        )

    def fit(self, X: ArrayLike, y: None = None) -> Self:
        if self.metric not in METRIC_MAP:
            raise ValueError(
                f"metric must be one of {sorted(METRIC_MAP)}, got {self.metric!r}"
            )
        if self.algorithm not in ("auto", "brute", "graph"):
            raise ValueError(
                f"algorithm must be 'auto', 'brute' or 'graph', got {self.algorithm!r}"
            )
        self.fit_X_ = cast(
            "NDArray[np.float32]", validate_data(self, X, dtype=np.float32)
        )
        self.n_samples_fit_ = self.fit_X_.shape[0]
        self._index = None
        return self

    def transform(self, X: ArrayLike) -> csr_matrix:
        X = cast(
            "NDArray[np.float32]",
            self._transform_checks(X, "fit_X_", dtype=np.float32),
        )
        k = min(self.n_neighbors + 1, self.n_samples_fit_)
        metric = self._metric_info.metric
        if self._exact():
            ids, dist = mlxvs.brute_force_knn(self.fit_X_, X, k, metric)
        else:
            ids, dist = self._graph_index().search(
                X, k, itopk_size=max(self.itopk_size, k)
            )
        return self._to_csr(np.asarray(ids), np.asarray(dist), X.shape[0])

    def fit_transform(self, X: ArrayLike, y: None = None) -> csr_matrix:  # type: ignore[override]
        self.fit(X)
        n = self.n_samples_fit_
        k = min(self.n_neighbors + 1, n)
        metric = self._metric_info.metric
        if self._exact() or k > 128:
            ids, dist = mlxvs.brute_force_knn(self.fit_X_, self.fit_X_, k, metric)
            ids, dist = np.asarray(ids), np.asarray(dist)
        else:
            # every row's k - 1 nearest other rows, then the row itself first
            others, odist = mlxvs.all_neighbors(self.fit_X_, k - 1, metric)
            ids = np.concatenate(
                [np.arange(n, dtype=np.int32)[:, None], np.asarray(others)], axis=1
            )
            self_dist = 1.0 if metric == "cosine" else 0.0
            dist = np.concatenate(
                [np.full((n, 1), self_dist, np.float32), np.asarray(odist)], axis=1
            )
        return self._to_csr(ids, dist, n, self_rows=True)

    def _graph_index(self) -> mlxvs.CagraIndex:
        index = self._index
        if index is None:
            degree = max(1, min(self.graph_degree, self.n_samples_fit_ - 1))
            index = mlxvs.CagraIndex.build(
                self.fit_X_,
                degree,
                self._metric_info.metric,
                intermediate_graph_degree=min(2 * degree, 128),
            )
            self._index = index
        return index

    def _to_csr(
        self,
        ids: NDArray[np.int32],
        dist: NDArray[np.float32],
        n_rows: int,
        *,
        self_rows: bool = False,
    ) -> csr_matrix:
        info = self._metric_info
        dist = dist.astype(np.float32)
        if info.negate:
            dist = 1.0 - dist
        dist = np.maximum(dist, 0.0)
        if info.sqrt:
            dist = np.sqrt(dist)
        if self_rows:
            # each row's own entry first and exactly 0, as KNeighborsTransformer
            rows = np.arange(n_rows)[:, None]
            own = ids == rows
            missing = ~own.any(axis=1)
            ids = ids.copy()
            ids[missing, -1] = rows[missing, 0]
            own = ids == rows
            order = np.argsort(~own, axis=1, kind="stable")
            ids = np.take_along_axis(ids, order, axis=1)
            dist = np.take_along_axis(dist, order, axis=1)
            dist[:, 0] = 0.0
        # rows with fewer neighbors than requested (-1 ids) keep only the found
        valid = ids >= 0
        indptr = np.concatenate([[0], np.cumsum(valid.sum(axis=1))])
        mat = csr_matrix(
            (dist[valid], ids[valid], indptr),
            shape=(n_rows, self.n_samples_fit_),
        )
        return postprocess_knn_csr(
            mat, include_fwd=self.include_fwd, include_rev=self.include_rev
        )

    def __getstate__(self) -> dict[str, object]:
        # the GPU index is rebuilt on demand
        state = self.__dict__.copy()
        state["_index"] = None
        return state

    def __sklearn_tags__(self) -> Tags:
        return Tags(
            estimator_type="transformer",
            target_tags=TargetTags(required=False),
            transformer_tags=TransformerTags(preserves_dtype=["float32"]),
        )
