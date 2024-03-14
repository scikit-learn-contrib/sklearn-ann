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

Installation
============

To install the latest release from PyPI, run:

.. code-block:: bash

    pip install sklearn-ann

To install the latest development version from GitHub, run:

.. code-block:: bash

    pip install git+https://github.com/scikit-learn-contrib/sklearn-ann.git#egg=sklearn-ann

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

This project is managed using Hatch_ and pre-commit_. To get started, run ``pre-commit
install`` and ``hatch env create``. Run all commands using ``hatch run python
<command>`` which will ensure the environment is kept up to date. pre-commit_ comes into
play on every `git commit` after installation.

Consult ``pyproject.toml`` for which dependency groups and extras exist,
and the Hatch help or user guide for more info on what they are.

.. _Hatch: https://hatch.pypa.io/
.. _pre-commit: https://pre-commit.com/
