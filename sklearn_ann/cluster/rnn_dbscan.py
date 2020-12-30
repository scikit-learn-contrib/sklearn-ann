from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.neighbors import KNeighborsTransformer
from collections import deque
from ..utils import get_sparse_row
import numpy as np


UNCLASSIFIED = -2
NOISE = -1


def join(it1, it2):
    cur_it1 = next(it1, None)
    cur_it2 = next(it2, None)
    while 1:
        if cur_it1 is None and cur_it2 is None:
            break
        elif cur_it1 is None:
            yield cur_it2
            cur_it2 = next(it2, None)
        elif cur_it2 is None:
            yield cur_it1
            cur_it1 = next(it1, None)
        elif cur_it1[0] == cur_it2[0]:
            yield cur_it1
            cur_it1 = next(it1, None)
            cur_it2 = next(it2, None)
        elif cur_it1[0] < cur_it2[0]:
            yield cur_it1
            cur_it1 = next(it1, None)
        else:
            yield cur_it2
            cur_it2 = next(it2, None)


def neighborhood(is_core, knns, rev_knns, idx):
    # TODO: Make this inner bit faster
    knn_it = get_sparse_row(knns, idx)
    rev_core_knn_it = (
        (other_idx, dist)
        for other_idx, dist in get_sparse_row(rev_knns, idx)
        if is_core[other_idx]
    )
    yield from (
        (other_idx, dist)
        for other_idx, dist in join(knn_it, rev_core_knn_it)
        if other_idx != idx
    )


def rnn_dbscan_inner(is_core, knns, rev_knns, labels):
    cluster = 0
    cur_dens = 0
    dens = []
    for x_idx in range(len(labels)):
        if labels[x_idx] == UNCLASSIFIED:
            # Expand cluster
            if is_core[x_idx]:
                labels[x_idx] = cluster
                # TODO: Make this inner bit faster - can just assume
                # sorted an keep sorted
                seeds = deque()
                for neighbor_idx, dist in neighborhood(is_core, knns, rev_knns, x_idx):
                    labels[neighbor_idx] = cluster
                    if dist > cur_dens:
                        cur_dens = dist
                    seeds.append(neighbor_idx)
                while seeds:
                    y_idx = seeds.popleft()
                    if is_core[y_idx]:
                        for z_idx, dist in neighborhood(is_core, knns, rev_knns, y_idx):
                            if dist > cur_dens:
                                cur_dens = dist
                            if labels[z_idx] == UNCLASSIFIED:
                                seeds.append(z_idx)
                                labels[z_idx] = cluster
                            elif labels[z_idx] == NOISE:
                                labels[z_idx] = cluster
                dens.append(cur_dens)
                cur_dens = 0
                cluster += 1
            else:
                labels[x_idx] = NOISE
    # Expand clusters
    for x_idx in range(len(labels)):
        if labels[x_idx] == NOISE:
            min_cluster = NOISE
            min_dist = float("inf")
            for n_idx, n_dist in get_sparse_row(knns, x_idx):
                if n_dist >= min_dist or not is_core[n_idx]:
                    continue
                cluster = labels[n_idx]
                if n_dist > dens[cluster]:
                    continue
                min_cluster = cluster
                min_dist = n_dist
            labels[x_idx] = min_cluster
    return dens


class RnnDBSCAN(ClusterMixin, BaseEstimator):
    def __init__(self, n_neighbors=5, *, input_guarantee="none", n_jobs=None):
        self.n_neighbors = n_neighbors
        self.input_guarantee = input_guarantee
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        X = self._validate_data(X, accept_sparse="csr")
        if self.input_guarantee == "none":
            algorithm = KNeighborsTransformer(n_neighbors=self.n_neighbors)
            X = algorithm.fit_transform(X)
        elif self.input_guarantee == "kneighbors":
            pass
        else:
            raise ValueError(
                "Expected input_guarantee to be one of 'none', 'kneighbors'"
            )
        import timeit
        XT = X.transpose().tocsr(copy=True)

        # Initially, all samples are unclassified.
        labels = np.full(X.shape[0], UNCLASSIFIED, dtype=np.int32)

        # A list of all core samples found. -1 is to account for diagonal.
        core_samples = XT.getnnz(1) - 1 >= self.n_neighbors

        dens = rnn_dbscan_inner(core_samples, X, XT, labels)

        self.core_sample_indices_ = core_samples.nonzero()
        self.labels_ = labels
        self.dens_ = dens

        return self

    def fit_predict(self, X, y=None):
        self.fit(X, y=y)
        return self.labels_


def simple_rnn_dbscan_pipeline(neighbor_transformer, n_neighbors, **kwargs):
    from sklearn.pipeline import make_pipeline
    n_jobs = kwargs.get("n_jobs", None)
    return make_pipeline(
        neighbor_transformer(
            n_neighbors=n_neighbors,
            **kwargs,
        ),
        RnnDBSCAN(
            n_neighbors=n_neighbors,
            input_guarantee="kneighbors",
            n_jobs=n_jobs
        ),
    )
