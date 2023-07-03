.. -*- mode: rst -*-

|ReadTheDocs|_

.. |ReadTheDocs| image:: https://readthedocs.org/projects/sklearn-ann/badge/?version=latest
.. _ReadTheDocs: https://sklearn-ann.readthedocs.io/en/latest/?badge=latest

sklearn-ann
===========

.. inclusion-marker-do-not-remove

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

Development
===========

This project is managed using Poetry_ and pre-commit_.
To get started, run ``pre-commit install`` once and ``poetry install ...``
whenever dependencies have changed. E.g. @flying-sheep runs::

    poetry install --with=test --extras=annlibs

This installs all optional (dev) dependencies except for those to build the docs.
pre-commit_ comes into play on every `git commit` after installation.

Consult ``pyproject.toml`` for which dependency groups and extras exist,
and the poetry help or user guide for more info on what they are.

.. _poetry: https://python-poetry.org/
.. _pre-commit: https://pre-commit.com/
