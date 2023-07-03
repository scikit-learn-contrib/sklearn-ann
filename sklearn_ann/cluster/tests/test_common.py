import pytest
from sklearn.utils.estimator_checks import check_estimator

from sklearn_ann.cluster.rnn_dbscan import RnnDBSCAN

ESTIMATORS = [RnnDBSCAN]


@pytest.mark.parametrize("Estimator", ESTIMATORS)
def test_all_estimators(Estimator):
    check_estimator(Estimator())
