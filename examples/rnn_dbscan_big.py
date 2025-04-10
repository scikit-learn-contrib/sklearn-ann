"""
=======================================================
Demo of RnnDBSCAN clustering algorithm on large dataset
=======================================================

Tests RnnDBSCAN on a large dataset. Requires pandas.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from joblib import Memory
from sklearn import metrics
from sklearn.datasets import fetch_openml

from sklearn_ann.cluster.rnn_dbscan import simple_rnn_dbscan_pipeline

if TYPE_CHECKING:
    from typing import Any

    from sklearn.utils import Bunch


# #############################################################################
# Generate sample data
def fetch_mnist() -> Bunch:
    print("Downloading mnist_784")
    mnist = fetch_openml("mnist_784")
    return mnist.data / 255, mnist.target


memory = Memory("./mnist")

X, y = memory.cache(fetch_mnist)()


def run_rnn_dbscan(
    neighbor_transformer: object, n_neighbors: int, **kwargs: Any
) -> None:
    # #############################################################################
    # Compute RnnDBSCAN

    pipeline = simple_rnn_dbscan_pipeline(neighbor_transformer, n_neighbors, **kwargs)
    labels = pipeline.fit_predict(X)
    db = pipeline.named_steps["rnndbscan"]
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print(f"""\
Estimated number of clusters: {n_clusters_}
Estimated number of noise points: {n_noise_}
Homogeneity: {metrics.homogeneity_score(y, labels):0.3f}
Completeness: {metrics.completeness_score(y, labels):0.3f}
V-measure: {metrics.v_measure_score(y, labels):0.3f}
Adjusted Rand Index: {metrics.adjusted_rand_score(y, labels):0.3f}
Adjusted Mutual Information: {metrics.adjusted_mutual_info_score(y, labels):0.3f}
Silhouette Coefficient: {metrics.silhouette_score(X, labels):0.3f}\
""")


if __name__ == "__main__":
    import code

    print("""\
Now you can import your chosen transformer_cls and run:
run_rnn_dbscan(transformer_cls, n_neighbors, **params)
e.g.
from sklearn_ann.kneighbors.pynndescent import PyNNDescentTransformer
run_rnn_dbscan(PyNNDescentTransformer, 10)\
""")
    code.interact(local=locals())
