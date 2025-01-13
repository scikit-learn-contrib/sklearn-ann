from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsTransformer
from sklearn.utils.estimator_checks import check_estimator

from sklearn_ann.test_utils import needs

if not TYPE_CHECKING:
    AnnoyTransformer = FAISSTransformer = NMSlibTransformer = None
    PyNNDescentTransformer = KNeighborsTransformer = None
if find_spec("annoy") or TYPE_CHECKING:
    from sklearn_ann.kneighbors.annoy import AnnoyTransformer
if find_spec("faiss") or TYPE_CHECKING:
    from sklearn_ann.kneighbors.faiss import FAISSTransformer
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
    | NMSlibTransformer
    | PyNNDescentTransformer
    | KNeighborsTransformer
)

ESTIMATORS: ParameterSet[Estimator] = [
    pytest.param(AnnoyTransformer, marks=[needs.annoy()]),
    pytest.param(FAISSTransformer, marks=[needs.faiss()]),
    pytest.param(NMSlibTransformer, marks=[needs.nmslib()]),
    pytest.param(PyNNDescentTransformer, marks=[needs.pynndescent()]),
    pytest.param(BallTreeTransformer),
    pytest.param(KDTreeTransformer),
]

PER_ESTIMATOR_XFAIL_CHECKS = {
    AnnoyTransformer: dict(check_estimators_pickle="Cannot pickle AnnoyIndex"),
    FAISSTransformer: dict(
        check_estimators_pickle="Cannot pickle FAISS index",
        check_methods_subset_invariance="Unable to reset FAISS internal RNG",
    ),
    NMSlibTransformer: dict(check_estimators_pickle="Cannot pickle NMSLib index"),
}


def add_mark(param: ParameterSet, mark: pytest.MarkDecorator) -> ParameterSet:
    return pytest.param(*param.values, marks=[*param.marks, mark], id=param.id)


@pytest.mark.parametrize(
    "Estimator",
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
        expected_failed_checks=PER_ESTIMATOR_XFAIL_CHECKS.get(Estimator, {}),
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


def mark_diagonal_0_xfail(est: ParameterSet[Estimator]) -> ParameterSet[Estimator]:
    """Mark flaky tests as xfail(strict=False)."""
    # Should probably postprocess these...
    reasons = {
        PyNNDescentTransformer: "sometimes doesn't return diagonal==0",
        FAISSTransformer: "sometimes returns diagonal==eps where eps is small",
    }
    [val] = est.values
    name = val.__name__ if isinstance(val, type) else val
    if reason := reasons.get(val):
        return add_mark(est, pytest.mark.xfail(reason=f"{name} {reason}", strict=False))
    return est


@pytest.mark.parametrize(
    "estim_cls", [mark_diagonal_0_xfail(est) for est in ESTIMATORS]
)
def test_all_return_diagonal_0(
    random_small: NDArray[np.float64], estim_cls: type[Estimator]
) -> None:
    # * only explicitly store nearest neighborhoods of each sample with respect to the
    #   training data. This should include those at 0 distance from a query point,
    #   including the matrix diagonal when computing the nearest neighborhoods
    #   between the training data and itself.

    # Check: do we alway get an "extra" neighbour (diagonal/self)
    est = estim_cls(n_neighbors=3)
    knns = est.fit_transform(random_small)
    assert (knns.getnnz(1) == 4).all()

    # Check: diagonal is 0
    next_expected_diagonal = 0
    for row_idx in range(knns.shape[0]):
        start_idx = knns.indptr[row_idx]
        end_idx = knns.indptr[row_idx + 1]
        for col_idx, val in zip(
            knns.indices[start_idx:end_idx], knns.data[start_idx:end_idx]
        ):
            print("self0", row_idx, start_idx, end_idx, col_idx, val)
            if row_idx != col_idx:
                continue
            assert col_idx == next_expected_diagonal
            assert val == 0
            next_expected_diagonal += 1
    assert next_expected_diagonal == len(random_small)


@pytest.mark.parametrize("Estimator", ESTIMATORS)
def test_all_same(estim_cls: type[Estimator]) -> None:
    # Again but for the case of the same element
    ones = np.ones((64, 4))
    est = estim_cls(n_neighbors=3)
    knns = est.fit_transform(ones)
    print("knns", knns)
    assert (knns.getnnz(1) == 4).all()
    assert len(knns.nonzero()[0]) == 0
