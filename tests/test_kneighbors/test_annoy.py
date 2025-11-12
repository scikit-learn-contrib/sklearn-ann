from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from sklearn_ann.test_utils import assert_row_close, needs

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray

try:
    from sklearn_ann.kneighbors.annoy import AnnoyTransformer
except ImportError:
    pass


@needs.annoy
def test_euclidean(
    random_small: NDArray[np.float64],
    random_small_pdists: Mapping[str, NDArray[np.float64]],
) -> None:
    trans = AnnoyTransformer(metric="euclidean")
    mat = trans.fit_transform(random_small)
    euclidean_dist = random_small_pdists["euclidean"]
    assert_row_close(mat, euclidean_dist)


@needs.annoy
@pytest.mark.xfail(reason="not sure why this isn't working")
def test_angular(
    random_small: NDArray[np.float64],
    random_small_pdists: Mapping[str, NDArray[np.float64]],
) -> None:
    trans = AnnoyTransformer(metric="angular")
    mat = trans.fit_transform(random_small)
    angular_dist = np.arccos(1 - random_small_pdists["cosine"])
    assert_row_close(mat, angular_dist)
