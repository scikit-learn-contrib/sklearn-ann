from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import annoy
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import Tags, TargetTags, TransformerTags
from sklearn.utils.validation import validate_data

from ..utils import TransformerChecksMixin, knn_csr

if TYPE_CHECKING:
    from typing import Self

    from numpy.typing import ArrayLike, NDArray
    from scipy.sparse import csr_array

#: The metrics :class:`AnnoyTransformer` supports.
type Metric = Literal[
    "angular", "euclidean", "manhattan", "hamming", "dot", "sqeuclidean"
]


class AnnoyTransformer(TransformerChecksMixin, TransformerMixin, BaseEstimator):
    """Wrapper for using annoy.AnnoyIndex as sklearn's KNeighborsTransformer"""

    def __init__(
        self,
        n_neighbors: int = 5,
        *,
        metric: Metric = "euclidean",
        n_trees: int = 10,
        search_k: int = -1,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.n_trees = n_trees
        self.search_k = search_k
        self.metric = metric

    def fit(self, X: ArrayLike, y: None = None) -> Self:
        X = cast("NDArray[np.float64]", validate_data(self, X))
        self.n_samples_fit_ = X.shape[0]
        metric = "euclidean" if self.metric == "sqeuclidean" else self.metric
        self.annoy_ = annoy.AnnoyIndex(X.shape[1], metric=metric)
        for i, x in enumerate(X):
            self.annoy_.add_item(i, x.tolist())
        self.annoy_.build(self.n_trees)
        return self

    def transform(self, X: NDArray[np.float64]) -> csr_array:
        X = self._transform_checks(X, "annoy_")
        return self._transform(X)

    def fit_transform(self, X: ArrayLike, y: None = None) -> csr_array:  # type: ignore[override]
        """Fit to `X` and return its kNN graph."""
        return self.fit(X)._transform(X=None)

    def _transform(self, X: NDArray[np.float64] | None) -> csr_array:
        """As `transform`, but handles X is None for faster `fit_transform`."""

        n_samples_transform = self.n_samples_fit_ if X is None else X.shape[0]

        # For compatibility reasons, as each sample is considered as its own
        # neighbor, one extra neighbor will be computed.
        n_neighbors = self.n_neighbors + 1

        # -1: the index found fewer neighbors
        indices = np.full((n_samples_transform, n_neighbors), -1)
        distances = np.zeros((n_samples_transform, n_neighbors))

        for i in range(n_samples_transform):
            ind, dist = (
                self.annoy_.get_nns_by_item(
                    i, n_neighbors, self.search_k, include_distances=True
                )
                if X is None
                else self.annoy_.get_nns_by_vector(
                    X[i].tolist(), n_neighbors, self.search_k, include_distances=True
                )
            )
            indices[i, : len(ind)], distances[i, : len(dist)] = ind, dist

        if self.metric == "sqeuclidean":
            distances **= 2

        return knn_csr(
            indices,
            distances,
            self.n_samples_fit_,
            self_rows=X is None,
            is_distance=self.metric != "dot",  # dot: similarities
        )

    def __sklearn_tags__(self) -> Tags:
        return Tags(
            estimator_type="transformer",
            target_tags=TargetTags(required=False),
            transformer_tags=TransformerTags(),
        )
