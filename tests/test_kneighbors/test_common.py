from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsTransformer
from sklearn.utils.estimator_checks import check_estimator

from sklearn_ann.test_utils import needs

if not TYPE_CHECKING:
    AnnoyTransformer = FAISSTransformer = MLXVSTransformer = NMSlibTransformer = None
    PyNNDescentTransformer = KNeighborsTransformer = None
if find_spec("annoy") or TYPE_CHECKING:
    from sklearn_ann.kneighbors.annoy import AnnoyTransformer
if find_spec("faiss") or TYPE_CHECKING:
    from sklearn_ann.kneighbors.faiss import FAISSTransformer
if find_spec("mlxvs") or TYPE_CHECKING:
    from sklearn_ann.kneighbors.mlxvs import MLXVSTransformer
if find_spec("nmslib") or TYPE_CHECKING:
    from sklearn_ann.kneighbors.nmslib import NMSlibTransformer
if find_spec("pynndescent") or TYPE_CHECKING:
    from sklearn_ann.kneighbors.pynndescent import PyNNDescentTransformer

from sklearn_ann.kneighbors.sklearn import BallTreeTransformer, KDTreeTransformer

if TYPE_CHECKING:
    from _pytest.mark import ParameterSet
    from numpy.typing import NDArray

Estimator = (
    AnnoyTransformer
    | FAISSTransformer
    | MLXVSTransformer
    | NMSlibTransformer
    | PyNNDescentTransformer
    | KNeighborsTransformer
)

ESTIMATORS: list[ParameterSet] = [
    pytest.param(AnnoyTransformer, marks=[needs.annoy()]),
    pytest.param(FAISSTransformer, marks=[needs.faiss()]),
    pytest.param(MLXVSTransformer, marks=[needs.mlxvs()]),
    pytest.param(NMSlibTransformer, marks=[needs.nmslib()]),
    pytest.param(PyNNDescentTransformer, marks=[needs.pynndescent()]),
    pytest.param(BallTreeTransformer),
    pytest.param(KDTreeTransformer),
]

PER_ESTIMATOR_XFAIL_CHECKS = {
    AnnoyTransformer: dict(
        check_estimators_pickle="Cannot pickle AnnoyIndex",
        # fit_transform uses get_nns_by_item, transform uses get_nns_by_vector
        check_transformer_general="fit_transform and transform disagree",
        check_transformer_data_not_an_array="fit_transform and transform disagree",
    ),
    FAISSTransformer: dict(
        check_estimators_pickle="Cannot pickle FAISS index",
        check_methods_subset_invariance="Unable to reset FAISS internal RNG",
    ),
    NMSlibTransformer: dict(check_estimators_pickle="Cannot pickle NMSLib index"),
}


def add_mark(param: ParameterSet, mark: pytest.MarkDecorator) -> ParameterSet:
    return pytest.param(*param.values, marks=[*param.marks, mark], id=param.id)


@pytest.mark.parametrize(
    "estim_cls",
    [
        add_mark(
            est,
            pytest.mark.xfail(
                reason="cannot deal with all dtypes (problem is upsteam)"
            ),
        )
        if est.values[0] is PyNNDescentTransformer
        else est
        for est in ESTIMATORS
    ],
)
def test_all_estimators(estim_cls: type[Estimator]) -> None:
    check_estimator(
        estim_cls(),
        expected_failed_checks=PER_ESTIMATOR_XFAIL_CHECKS.get(estim_cls, {}),
    )


# The following critera are from:
#   https://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbors-transformer
# * only explicitly store nearest neighborhoods of each sample with respect to the
#   training data. This should include those at 0 distance from a query point,
#   including the matrix diagonal when computing the nearest neighborhoods between the
#   training data and itself.
# * each row’s data should store the distance in increasing order
#   (optional. Unsorted data will be stable-sorted, adding a computational overhead).
# * all values in data should be non-negative.
# * there should be no duplicate indices in any row (see https://github.com/scipy/scipy/issues/5807).
# * if the algorithm being passed the precomputed matrix uses k nearest neighbors
#   (as opposed to radius neighborhood), at least k neighbors must be stored in each row
#   (or k+1, as explained in the following note).


@pytest.mark.parametrize("estim_cls", ESTIMATORS)
def test_all_follow_knn_spec(
    random_small: NDArray[np.float64], estim_cls: type[Estimator]
) -> None:
    knns = estim_cls(n_neighbors=3).fit_transform(random_small)
    # an "extra" neighbor: the row itself
    assert (np.diff(knns.indptr) == 4).all()
    # it is the first neighbor and at distance 0
    first = knns.indptr[:-1]
    np.testing.assert_array_equal(knns.indices[first], np.arange(knns.shape[0]))
    np.testing.assert_array_equal(knns.data[first], 0)
    assert (knns.data >= 0).all()
    for start, end in zip(knns.indptr[:-1], knns.indptr[1:]):
        assert len(set(knns.indices[start:end])) == end - start, "duplicate indices"
        assert (np.diff(knns.data[start:end]) >= 0).all(), "unsorted distances"


@pytest.mark.parametrize("estim_cls", ESTIMATORS)
def test_all_same(estim_cls: type[Estimator]) -> None:
    # Again but for the case of the same element
    ones = np.ones((64, 4))
    est = estim_cls(n_neighbors=3)
    knns = est.fit_transform(ones)
    print("knns", knns)
    assert (np.diff(knns.indptr) == 4).all()
    assert len(knns.nonzero()[0]) == 0
