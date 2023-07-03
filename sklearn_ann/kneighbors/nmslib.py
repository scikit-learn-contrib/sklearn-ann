import nmslib
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin

from ..utils import TransformerChecksMixin, check_metric

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
        self, n_neighbors=5, *, metric="euclidean", method="sw-graph", n_jobs=1
    ):
        self.n_neighbors = n_neighbors
        self.method = method
        self.metric = metric
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        X = self._validate_data(X)
        self.n_samples_fit_ = X.shape[0]

        check_metric(self.metric, METRIC_MAP)
        space = METRIC_MAP[self.metric]

        self.nmslib_ = nmslib.init(method=self.method, space=space)
        self.nmslib_.addDataPointBatch(X)
        self.nmslib_.createIndex()
        return self

    def transform(self, X):
        X = self._transform_checks(X, "nmslib_")
        n_samples_transform = X.shape[0]

        # For compatibility reasons, as each sample is considered as its own
        # neighbor, one extra neighbor will be computed.
        n_neighbors = self.n_neighbors + 1

        results = self.nmslib_.knnQueryBatch(X, k=n_neighbors, num_threads=self.n_jobs)
        indices, distances = zip(*results)
        indices, distances = np.vstack(indices), np.vstack(distances)

        if self.metric == "sqeuclidean":
            distances **= 2

        indptr = np.arange(0, n_samples_transform * n_neighbors + 1, n_neighbors)
        kneighbors_graph = csr_matrix(
            (distances.ravel(), indices.ravel(), indptr),
            shape=(n_samples_transform, self.n_samples_fit_),
        )

        return kneighbors_graph

    def _more_tags(self):
        return {
            "_xfail_checks": {"check_estimators_pickle": "Cannot pickle NMSLib index"},
            "preserves_dtype": [np.float32],
        }
