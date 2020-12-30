from ..faiss import FAISSTransformer
from .utils import assert_row_close


def test_euclidean(random_small, random_small_pdists):
    trans = FAISSTransformer(metric="euclidean")
    mat = trans.fit_transform(random_small)
    euclidean_dist = random_small_pdists["euclidean"]
    assert_row_close(mat, euclidean_dist)


def test_cosine(random_small, random_small_pdists):
    trans = FAISSTransformer(metric="cosine")
    mat = trans.fit_transform(random_small)
    cosine_dist = random_small_pdists["cosine"]
    assert_row_close(mat, cosine_dist)
