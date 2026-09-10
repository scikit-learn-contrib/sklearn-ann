from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sklearn.utils.estimator_checks import check_estimator

from sklearn_ann.cluster.rnn_dbscan import RnnDBSCAN

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator


ESTIMATORS: list[type[BaseEstimator]] = [RnnDBSCAN]


@pytest.mark.parametrize("Estimator", ESTIMATORS)
def test_all_estimators(Estimator: type[BaseEstimator]) -> None:
    check_estimator(Estimator())
