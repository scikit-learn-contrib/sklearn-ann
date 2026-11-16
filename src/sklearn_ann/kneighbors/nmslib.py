from __future__ import annotations

from typing import TYPE_CHECKING, cast

import nmslib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import Tags, TargetTags, TransformerTags
from sklearn.utils.validation import validate_data

from ..utils import TransformerChecksMixin, check_metric, knn_csr

if TYPE_CHECKING:
    from typing import Self

    from numpy.typing import ArrayLike, NDArray
    from scipy.sparse import csr_matrix

# see more metric in the manual
# https://github.com/nmslib/nmslib/tree/master/manual
METRIC_MAP = {
    "sqeuclidean": "l2",
    "euclidean": "l2",
    "cosine": "cosinesimil",
    "l1": "l1",
    "l2": "l2",
}


class NMSlibTransformer(TransformerChecksMixin, TransformerMixin, BaseEstimator):
    """Wrapper for using nmslib as sklearn's KNeighborsTransformer"""

    def __init__(
        self,
        n_neighbors: int = 5,
        *,
        metric: str = "euclidean",
        method: str = "sw-graph",
        n_jobs: int = 1,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.method = method
        self.metric = metric
        self.n_jobs = n_jobs

    def fit(self, X: ArrayLike, y: None = None) -> Self:
        X = cast("NDArray[np.float64]", validate_data(self, X))
        self.n_samples_fit_ = X.shape[0]

        check_metric(self.metric, METRIC_MAP)
        space = METRIC_MAP[self.metric]

        self.nmslib_ = nmslib.init(method=self.method, space=space)
        self.nmslib_.addDataPointBatch(X)
        self.nmslib_.createIndex()
        return self

    def transform(self, X: NDArray[np.float64]) -> csr_matrix:
        X = self._transform_checks(X, "nmslib_")
        return self._transform(X)

    def fit_transform(self, X: ArrayLike, y: None = None) -> csr_matrix:  # type: ignore[override]
        X = cast("NDArray[np.float64]", validate_data(self, X))
        return self.fit(X)._transform(X, self_rows=True)

    def _transform(
        self, X: NDArray[np.float64], *, self_rows: bool = False
    ) -> csr_matrix:
        # For compatibility reasons, as each sample is considered as its own
        # neighbor, one extra neighbor will be computed.
        n_neighbors = self.n_neighbors + 1

        results = self.nmslib_.knnQueryBatch(X, k=n_neighbors, num_threads=self.n_jobs)
        # -1: the index found fewer neighbors
        indices = np.full((X.shape[0], n_neighbors), -1)
        distances = np.zeros((X.shape[0], n_neighbors), dtype=np.float32)
        for i, (ind, dist) in enumerate(results):
            indices[i, : len(ind)], distances[i, : len(dist)] = ind, dist

        if self.metric == "sqeuclidean":
            distances **= 2

        return knn_csr(indices, distances, self.n_samples_fit_, self_rows=self_rows)

    def __sklearn_tags__(self) -> Tags:
        return Tags(
            estimator_type="transformer",
            target_tags=TargetTags(required=False),
            transformer_tags=TransformerTags(preserves_dtype=["float32"]),
        )
