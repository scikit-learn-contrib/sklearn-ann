"""
===================================
Demo of RNN-DBSCAN clustering algorithm
===================================

Finds core samples of high density and expands clusters from them.

Mostly copypasted from sklearn's DBSCAN example.

"""

import numpy as np
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

from sklearn_ann.cluster.rnn_dbscan import RnnDBSCAN

# #############################################################################
# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(
    n_samples=750, centers=centers, cluster_std=0.4, random_state=0
)

X = StandardScaler().fit_transform(X)

# #############################################################################
# Compute DBSCAN
db = RnnDBSCAN(n_neighbors=10).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print(f"""\
Estimated number of clusters: {n_clusters_}
Estimated number of noise points: {n_noise_}
Homogeneity: {metrics.homogeneity_score(labels_true, labels):0.3f}
Completeness: {metrics.completeness_score(labels_true, labels):0.3f}
V-measure: {metrics.v_measure_score(labels_true, labels):0.3f}
Adjusted Rand Index: {metrics.adjusted_rand_score(labels_true, labels):0.3f}
Adjusted Mutual Info: {metrics.adjusted_mutual_info_score(labels_true, labels):0.3f}
Silhouette Coefficient: {metrics.silhouette_score(X, labels):0.3f}\
""")

# #############################################################################
# Plot result
import matplotlib.pyplot as plt

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = labels == k

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=14,
    )

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=6,
    )

plt.title("Estimated number of clusters: %d" % n_clusters_)
plt.show()
