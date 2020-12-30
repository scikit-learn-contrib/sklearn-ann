import pytest
from numpy.random import default_rng
from scipy.spatial.distance import pdist, squareform


@pytest.fixture(scope="module")
def random_small():
    gen = default_rng(42)
    return 2 * gen.random((64, 128)) - 1


@pytest.fixture(scope="module")
def random_small_pdists(random_small):
    return {
        metric: squareform(pdist(random_small, metric=metric))
        for metric in ["euclidean", "cosine"]
    }
