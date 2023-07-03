"""
=======================================================
Demo of RnnDBSCAN clustering algorithm on large dataset
=======================================================

Tests RnnDBSCAN on a large dataset. Requires pandas.

"""
print(__doc__)

import numpy as np
from joblib import Memory
from sklearn import metrics
from sklearn.datasets import fetch_openml

from sklearn_ann.cluster.rnn_dbscan import simple_rnn_dbscan_pipeline


# #############################################################################
# Generate sample data
def fetch_mnist():
    print("Downloading mnist_784")
    mnist = fetch_openml("mnist_784")
    return mnist.data / 255, mnist.target


memory = Memory("./mnist")

X, y = memory.cache(fetch_mnist)()


def run_rnn_dbscan(neighbor_transformer, n_neighbors, **kwargs):
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

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(y, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(y, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(y, labels))
    print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(y, labels))
    print(
        "Adjusted Mutual Information: %0.3f"
        % metrics.adjusted_mutual_info_score(y, labels)
    )
    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))


if __name__ == "__main__":
    import code

    print("Now you can import your chosen transformer_cls and run:")
    print("run_rnn_dbscan(transformer_cls, n_neighbors, **params)")
    print("e.g.")
    print("from sklearn_ann.kneighbors.pynndescent import PyNNDescentTransformer")
    print("run_rnn_dbscan(PyNNDescentTransformer, 10)")
    code.interact(local=locals())
