# hannoy needs a filesystem path (LMDB-backed)
import tempfile

import hannoy
import numpy as np
from hannoy import Metric
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import Tags, TargetTags, TransformerTags
from sklearn.utils.validation import validate_data

from ..utils import TransformerChecksMixin

# what other metrics should I include?
# it has cosine, manhattan, hamming and quantize counterparts:
# like binary quantitized consine, bq euclidean, and bq manhattan
METRIC_MAP = {
    "euclidean": Metric.EUCLIDEAN,
}

class HannoyTransformer(TransformerChecksMixin, TransformerMixin, BaseEstimator):
    # known issue where multiple Database instances silently share the first one's LMDB env

    def __init__( self, n_neighbors=5, *, metric="euclidean", path=None, m=16,
                 ef_construction=96, ef_search=200):
        self.n_neighbors = n_neighbors
        self.metric = metric
        # LMDB directory for the index; if None = auto-create a temp dir
        self.path = path
        # edges per node in the HNSW graph; hannoy default is 16
        self.m = m
        # hannoy default is 96 (higher = better graph but slower build)
        self.ef_construction = ef_construction
        # hannoy default is 200 (higher = better recall, slower search)
        self.ef_search = ef_search

    def fit(self, X, y = None):
        X = validate_data(self, X)
        self.n_samples_fit_ = X.shape[0]

        # storing validated X for fit_transform as hannoy doesn't have by_item yet
        self.fit_X = X
        # path to LMDB
        path = self.path if self.path is not None else tempfile.mkdtemp(prefix="hannoy_")

        # converting to the metric names used by hannoy
        hannoy_metric = METRIC_MAP[self.metric]

        # metric is fixed for the entire database
        self.hannoy_db_ = hannoy.Database(path, hannoy_metric)
        with self.hannoy_db_.writer(
            X.shape[1], m=self.m, ef=self.ef_construction
        ) as writer:
            for i, x in enumerate(X):
                # convert to list as hannoy's Rust code expects that type
                writer.add_item(i, x.tolist())
        # opening a reader query
        self.hannoy_reader_ = self.hannoy_db_.reader()
        return self

    def transform(self, X):
        # verify that fit was called and + that X has the right number of features
        X = self._transform_checks(X, "hannoy_reader_")
        return self._transform(X)

    def _transform(self, X):
        # how many points
        n_samples_transform = X.shape[0]

        n_neighbors = self.n_neighbors + 1
        # pre allocating indicies for which points are neighbots
        # distances = how far away
        # ELLPACk (similar)
        indices = np.empty((n_samples_transform, n_neighbors), dtype=int)
        distances = np.empty((n_samples_transform, n_neighbors))

        for i, x in enumerate(X):
            # hannoy for each row by_vec
            results = self.hannoy_reader_.by_vec(
                # returning in a list form because Rust requires it
                x.tolist(), n=n_neighbors, ef_search=self.ef_search
            )
            # unpacking into pre-allocated arrays
            for j, (idx, dist) in enumerate(results):
                indices[i, j] = idx
                distances[i, j] = dist
        # distance correction
        if self.metric == "euclidean":
            np.sqrt(distances, out=distances)

        # going from ELLPACK-like structure into CSR matrix
        indptr = np.arange(0, n_samples_transform * n_neighbors + 1, n_neighbors)
        kneighbors_graph = csr_matrix(
            (distances.ravel(), indices.ravel(), indptr),
            shape=(n_samples_transform, self.n_samples_fit_),
        )

        return kneighbors_graph

    def fit_transform(self, X, y=None):
        self.fit(X)
        result = self._transform(self.fit_X)
        # don't need those vectors anymore, so delete it
        del self.fit_X
        return result

    def __sklearn_tags__(self) -> Tags:
        # metadata
        return Tags(
            estimator_type="transformer",
            target_tags=TargetTags(required=False),
            transformer_tags=TransformerTags(),
        )
