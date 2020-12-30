.. -*- mode: rst -*-

|ReadTheDocs|_

.. |ReadTheDocs| image:: https://readthedocs.org/projects/sklearn-template/badge/?version=latest
.. _ReadTheDocs: https://sklearn-template.readthedocs.io/en/latest/?badge=latest

sklearn-ann
===========

**sklearn-ann** eases integration of approximate nearest neighbours
libraries such as annoy, nmslib and faiss into your sklearn
pipelines. It consists of:

* ``Transformers`` conforming to the same interface as
  ``KNeighborsTransformer`` which can be used to transform feature matrices
  into sparse distance matrices for use by any estimator that can deal with
  sparse distance matrices. Many, but not all, of scikit-learn's clustering and
  manifold learning algorithms can work with this kind of input.
* RNN-DBSCAN: a variant of DBSCAN based on reverse nearest
  neighbours.

Why? When do I want this?
=========================

The main scenarios in which this is needed is for performing
*clustering or manifold learning or high dimensional data*. The
reason is that currently the only neighbourhood algorithms which are
build into scikit-learn are essentially the standard tree approaches
to space partitioning: the ball tree and the K-D tree. These do not
perform competitively in high dimensional spaces.
