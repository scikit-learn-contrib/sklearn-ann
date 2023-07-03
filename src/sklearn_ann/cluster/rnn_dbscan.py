from collections import deque

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.neighbors import KNeighborsTransformer

from ..utils import get_sparse_row

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
    """
    Implements the RNN-DBSCAN clustering algorithm.

    Parameters
    ----------
    n_neighbors : int
        The number of neighbors in the kNN-graph (the k in kNN), and the
        theshold of reverse nearest neighbors for a node to be considered a
        core node.
    input_guarantee : "none" | "kneighbors"
        A guarantee on input matrices. If equal to "kneighbors", the algorithm
        will assume you are passing in the kNN graph exactly as required, e.g.
        with n_neighbors. This can be used to pass in a graph produced by one
        of the implementations of the KNeighborsTransformer interface.
    n_jobs : int
        The number of jobs to use. Currently has not effect since no part of
        the algorithm has been parallelled.
    keep_knns : bool
        If true, the kNN and inverse kNN graph will be saved to `knns_` and
        `rev_knns_` after fitting.

    See Also
    --------
    simple_rnn_dbscan_pipeline:
        Create a pipeline of a KNeighborsTransformer and RnnDBSCAN

    References
    ----------
    A. Bryant and K. Cios, "RNN-DBSCAN: A Density-Based Clustering
    Algorithm Using Reverse Nearest Neighbor Density Estimates," in IEEE
    Transactions on Knowledge and Data Engineering, vol. 30, no. 6, pp.
    1109-1121, 1 June 2018, doi: 10.1109/TKDE.2017.2787640.
    """

    def __init__(
        self, n_neighbors=5, *, input_guarantee="none", n_jobs=None, keep_knns=False
    ):
        self.n_neighbors = n_neighbors
        self.input_guarantee = input_guarantee
        self.n_jobs = n_jobs
        self.keep_knns = keep_knns

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

        XT = X.transpose().tocsr(copy=True)
        if self.keep_knns:
            self.knns_ = X
            self.rev_knns_ = XT

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

    def drop_knns(self):
        del self.knns_
        del self.rev_knns_


def simple_rnn_dbscan_pipeline(
    neighbor_transformer, n_neighbors, n_jobs=None, keep_knns=None, **kwargs
):
    """
    Create a simple pipeline comprising a transformer and RnnDBSCAN.

    Parameters
    ----------
    neighbor_transformer : class implementing KNeighborsTransformer interface
    n_neighbors:
        Passed to neighbor_transformer and RnnDBSCAN
    n_jobs:
        Passed to neighbor_transformer and RnnDBSCAN
    keep_knns:
        Passed to RnnDBSCAN
    kwargs:
        Passed to neighbor_transformer
    """
    from sklearn.pipeline import make_pipeline

    return make_pipeline(
        neighbor_transformer(n_neighbors=n_neighbors, n_jobs=n_jobs, **kwargs),
        RnnDBSCAN(
            n_neighbors=n_neighbors,
            input_guarantee="kneighbors",
            n_jobs=n_jobs,
            keep_knns=keep_knns,
        ),
    )
