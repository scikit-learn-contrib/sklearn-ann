from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING

import numpy as np

from sklearn_ann.test_utils import assert_row_close, needs

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray

if TYPE_CHECKING or find_spec("hannoy"):
    from hannoy import Metric

    from sklearn_ann.kneighbors.hannoy import HannoyTransformer


@needs.hannoy
def test_euclidean(
    random_small: NDArray[np.float64],
    random_small_pdists: Mapping[str, NDArray[np.float64]],
) -> None:
    trans = HannoyTransformer(metric=Metric.EUCLIDEAN)
    mat = trans.fit_transform(random_small)
    euclidean_dist = random_small_pdists["euclidean"]
    assert_row_close(mat, euclidean_dist)


@needs.hannoy
def test_cosine(
    random_small: NDArray[np.float64],
    random_small_pdists: Mapping[str, NDArray[np.float64]],
) -> None:
    trans = HannoyTransformer(metric=Metric.COSINE)
    mat = trans.fit_transform(random_small)
    # hannoy's cosine metric returns (1 - cos_sim) / 2
    cosine_dist = random_small_pdists["cosine"] / 2
    assert_row_close(mat, cosine_dist)


@needs.hannoy
def test_transform_matches_fit_transform(random_small: NDArray[np.float64]) -> None:
    # both entry points go through by_array on the same vectors
    # they should agree exactly;
    trans = HannoyTransformer(metric=Metric.EUCLIDEAN)
    by_item = trans.fit_transform(random_small)
    by_vec = trans.fit(random_small).transform(random_small)
    by_item.sort_indices()
    by_vec.sort_indices()
    np.testing.assert_array_equal(by_item.indices, by_vec.indices)
    np.testing.assert_allclose(by_item.data, by_vec.data, atol=1e-5)
