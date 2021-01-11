Implementations of the KNeighborsTransformer interface
======================================================

This module contains transformers which transform from array-like structures of
shape (n_samples, n_features) to KNN-graphs encoded as scipy.sparse.csr_matrix.
They conform to the KNeighborsTransformer interface. Each submodule in this
module provides facilities for exactly one external nearest neighbour library.

Annoy
-----

`Annoy (Approximate Nearest Neighbors Oh Yeah)`_ is a C++ library with Python
bindings to search for points in space that are close to a given query point. The originates from Spotify.
It uses a forest of random projection trees.

.. _`Annoy (Approximate Nearest Neighbors Oh Yeah)`: https://github.com/spotify/annoy


.. automodule:: sklearn_ann.kneighbors.annoy
   :members:

FAISS
-----

`FAISS (Facebook AI Similarity Search)`_ is a library for efficient similarity
search and clustering of dense vectors. The project originates from Facebook AI
Research (FAIR). It contains multiple algorithms including algorithms for
exact/brute force nearest neighbour, methods based on quantization and product
quantization, and methods based on Hierarchical Navigable Small World graphs
(HNSW). There are some `guidelines on how to choose the best index for your
purposes`.

.. _`FAISS (Facebook AI Similarity Search)`: https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index

.. _`guidelines on how to choose the best index for your purposes`: https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index


.. automodule:: sklearn_ann.kneighbors.faiss
   :members:

nmslib
------

`nmslib (non-metric space library)` is a library for similarity search support
metric and non-metric spaces. It contains multiple algorithms.


.. automodule:: sklearn_ann.kneighbors.nmslib
   :members:

PyNNDescent
-----------

`PyNNDescent`_ is a Python nearest neighbor descent for approximate nearest
neighbors. It iteratively improves kNN-graph using the transitive property,
using random projections for initialisation. This transformer is actually
implemented as part of PyNNDescent, and simply re-exported here for (foolish)
consistency. If you only need this transformer, just use PyNNDescent directly.


.. automodule:: sklearn_ann.kneighbors.pynndescent
   :members:

sklearn
-------

`scikit-learn` itself contains ball tree and k-d indices. KNeighborsTransformer is re-exported here specialised for these two types of index for consistency.


.. automodule:: sklearn_ann.kneighbors.sklearn
   :members:
