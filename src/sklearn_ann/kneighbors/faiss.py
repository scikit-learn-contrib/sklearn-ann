from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple, cast

import faiss
import numpy as np
from faiss import normalize_L2
from joblib import cpu_count
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import Tags, TargetTags, TransformerTags
from sklearn.utils.validation import validate_data

from ..utils import TransformerChecksMixin, postprocess_knn_csr

if TYPE_CHECKING:
    from typing import Self

    from numpy.typing import ArrayLike, NDArray


class MetricInfo(NamedTuple):
    metric: int
    normalize: bool = False
    negate: bool = False
    sqrt: bool = False


L2_INFO = MetricInfo(faiss.METRIC_L2, sqrt=True)


METRIC_MAP: dict[str, MetricInfo] = {
    "cosine": MetricInfo(faiss.METRIC_INNER_PRODUCT, normalize=True, negate=True),
    "l1": MetricInfo(faiss.METRIC_L1),
    "cityblock": MetricInfo(faiss.METRIC_L1),
    "manhattan": MetricInfo(faiss.METRIC_L1),
    "l2": L2_INFO,
    "euclidean": L2_INFO,
    "sqeuclidean": MetricInfo(faiss.METRIC_L2),
    "canberra": MetricInfo(faiss.METRIC_Canberra),
    "braycurtis": MetricInfo(faiss.METRIC_BrayCurtis),
    "jensenshannon": MetricInfo(faiss.METRIC_JensenShannon),
}


def mk_faiss_index(
    feats: NDArray[np.float32],
    inner_metric: int,
    index_key: str = "",
    nprobe: int = 128,
) -> faiss.Index:
    size, dim = feats.shape
    index: faiss.Index
    if not index_key:
        if inner_metric == faiss.METRIC_INNER_PRODUCT:
            index = faiss.IndexFlatIP(dim)
        else:
            index = faiss.IndexFlatL2(dim)
    else:
        if index_key.find("HNSW") < 0:
            raise NotImplementedError(
                "HNSW not implemented: returns distances insted of sims"
            )
        nlist = min(4096, 8 * round(math.sqrt(size)))
        index = faiss.index_factory(dim, index_key, inner_metric)
        if index_key.find("Flat") < 0:
            assert not index.is_trained
        index.train(feats)
        if isinstance(index, faiss.IndexIVF):
            index.nprobe = min(nprobe, nlist)
        assert index.is_trained
    index.add(feats)
    return index


class FAISSTransformer(TransformerChecksMixin, TransformerMixin, BaseEstimator):
    def __init__(
        self,
        n_neighbors: int = 5,
        *,
        metric: str = "euclidean",
        index_key: str = "",
        n_probe: int = 128,
        n_jobs: int = -1,
        include_fwd: bool = True,
        include_rev: bool = False,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.index_key = index_key
        self.n_probe = n_probe
        self.n_jobs = n_jobs
        self.include_fwd = include_fwd
        self.include_rev = include_rev

    @property
    def _metric_info(self) -> MetricInfo:
        return METRIC_MAP[self.metric]

    def fit(self, X: ArrayLike, y: None = None) -> Self:
        normalize = self._metric_info.normalize
        X = cast(
            "NDArray[np.float32]",
            validate_data(self, X, dtype=np.float32, copy=normalize),
        )
        self.n_samples_fit_ = X.shape[0]
        if self.n_jobs == -1:
            n_jobs = cpu_count()
        else:
            n_jobs = self.n_jobs
        faiss.omp_set_num_threads(n_jobs)
        if normalize:
            normalize_L2(X)
        self.faiss_ = mk_faiss_index(
            X, self._metric_info.metric, self.index_key, self.n_probe
        )
        return self

    def transform(self, X: NDArray[np.number]) -> csr_matrix:
        normalize = self._metric_info.normalize
        X = cast(
            "NDArray[np.float32]",
            self._transform_checks(X, "faiss_", dtype=np.float32, copy=normalize),
        )
        if normalize:
            normalize_L2(X)
        return self._transform(X)

    def _transform(self, X: NDArray[np.float32] | None) -> csr_matrix:
        n_samples_transform = self.n_samples_fit_ if X is None else X.shape[0]
        n_neighbors = self.n_neighbors + 1
        if X is None:
            # only flat indices store their vectors, i.e. `index_key` has to be empty
            assert isinstance(self.faiss_, faiss.IndexFlat)
            sims, nbrs = self.faiss_.search(
                np.reshape(
                    faiss.rev_swig_ptr(
                        self.faiss_.get_xb(), self.faiss_.ntotal * self.faiss_.d
                    ),
                    (self.faiss_.ntotal, self.faiss_.d),
                ),
                k=n_neighbors,
            )
        else:
            sims, nbrs = self.faiss_.search(X, k=n_neighbors)
        dist_arr = np.array(sims, dtype=np.float32)
        if self._metric_info.sqrt:
            dist_arr = np.sqrt(dist_arr)
        if self._metric_info.negate:
            dist_arr = 1 - dist_arr
        del sims
        nbr_arr = np.array(nbrs, dtype=np.int32)
        del nbrs
        indptr = np.arange(0, n_samples_transform * n_neighbors + 1, n_neighbors)
        """
        dist_arr = np.concatenate(
            [
                np.zeros(
                    (n_samples_transform, 1),
                    dtype=dist_arr.dtype
                ),
                dist_arr
            ], axis=1
        )
        nbr_arr = np.concatenate(
            [
                np.arange(n_samples_transform)[:, np.newaxis],
                nbr_arr
            ], axis=1
        )
        """
        mat = csr_matrix(
            (dist_arr.ravel(), nbr_arr.ravel(), indptr),
            shape=(n_samples_transform, self.n_samples_fit_),
        )
        return postprocess_knn_csr(
            mat, include_fwd=self.include_fwd, include_rev=self.include_rev
        )

    def fit_transform(self, X: ArrayLike, y: None = None) -> csr_matrix:  # type: ignore[override]
        return self.fit(X, y=y)._transform(X=None)

    def __sklearn_tags__(self) -> Tags:
        return Tags(
            estimator_type="transformer",
            target_tags=TargetTags(required=False),
            transformer_tags=TransformerTags(preserves_dtype=["float32"]),
            # Could be made deterministic *if* we could reset FAISS's internal RNG
            non_deterministic=True,
        )
