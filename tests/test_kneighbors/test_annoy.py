import numpy as np
import pytest

from sklearn_ann.test_utils import assert_row_close, needs

try:
    from sklearn_ann.kneighbors.annoy import AnnoyTransformer
except ImportError:
    pass


@needs.annoy
def test_euclidean(random_small, random_small_pdists):
    trans = AnnoyTransformer(metric="euclidean")
    mat = trans.fit_transform(random_small)
    euclidean_dist = random_small_pdists["euclidean"]
    assert_row_close(mat, euclidean_dist)


@needs.annoy
@pytest.mark.xfail(reason="not sure why this isn't working")
def test_angular(random_small, random_small_pdists):
    trans = AnnoyTransformer(metric="angular")
    mat = trans.fit_transform(random_small)
    angular_dist = np.arccos(1 - random_small_pdists["cosine"])
    assert_row_close(mat, angular_dist)
