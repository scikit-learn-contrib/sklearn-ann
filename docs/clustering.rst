Clustering
==========

While it is possible to use the transformers of the sklearn_ann.kneighbors module together with clustering algorithms from scikit-learn directly, there is often a mismatch between techniques like DBSCAN, which require for each node its neighbors within a certain radius, and kNN-graph which has a fixed number of. This mismatch may result in k being set to high, to make sure that, slowing things down.

This module contains an implementation of RNN-DBSCAN, which is based on the kNN-graph structure.

.. automodule:: sklearn_ann.cluster.rnn_dbscan
   :members:
