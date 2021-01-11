Background and API design
=========================

There have been long standing efficiency issues with scikit-learn's. In
particular, the `ball tree`_ and `k-d tree`_ to not scale well to high
dimensional spaces. The decision was taken that the best way to integrate other
techniques was to allow all applicable unsupervised estimators methods to take
a sparse matrix, typically being a KNN-graph of the points, but potentially
being any estimate. These `slides from PyParis 2018`_ explain some background,
while `issue #10463`_ and `pull request #10482`_ give discussion, justification
and benchmarks and more detail regarding the approach.

The main advantage of this technique is that the sparse matrix/KNN-graph can be built transformer from the data, and these to be sequenced using the scikit-learn pipeline mechanism. This approach allows for, for example parameter search to be done on the KNN-graph construction technique together with the estimator. Typically the transformer should closely follow the interface of KNeighborsTransformer. The `exact contract is outlined in the user guide`_.  .  There is also `an example notebook with early versions of the transformers in this library`_.

.. _`ball tree`: https://en.wikipedia.org/wiki/Ball_tree
.. _`k-d tree`: https://en.wikipedia.org/wiki/K-d_tree
.. _`slides from PyParis 2018`: https://tomdlt.github.io/decks/2018_pyparis/
.. _`issue #10463`: https://github.com/scikit-learn/scikit-learn/issues/10463
.. _`pull request #10482`: https://github.com/scikit-learn/scikit-learn/pull/10482
.. _`exact contract is outlined in the user guide`: https://scikit-learn.org/stable/modules/neighbors.html#neighbors-transformer
.. _`an example notebook with early versions of the transformers in this library`: https://scikit-learn.org/stable/auto_examples/neighbors/approximate_nearest_neighbors.html#sphx-glr-auto-examples-neighbors-approximate-nearest-neighbors-py
