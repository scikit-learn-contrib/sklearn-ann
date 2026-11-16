from __future__ import annotations

from typing import TYPE_CHECKING

from sklearn_ann.test_utils import assert_row_close, needs

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy as np
    from numpy.typing import NDArray

try:
    from sklearn_ann.kneighbors.nmslib import NMSlibTransformer
except ImportError:
    pass


@needs.nmslib
def test_euclidean(
    random_small: NDArray[np.float64],
    random_small_pdists: Mapping[str, NDArray[np.float64]],
) -> None:
    trans = NMSlibTransformer(metric="euclidean")
    mat = trans.fit_transform(random_small)
    euclidean_dist = random_small_pdists["euclidean"]
    assert_row_close(mat, euclidean_dist)


@needs.nmslib
def test_cosine(
    random_small: NDArray[np.float64],
    random_small_pdists: Mapping[str, NDArray[np.float64]],
) -> None:
    trans = NMSlibTransformer(metric="cosine")
    mat = trans.fit_transform(random_small)
    cosine_dist = random_small_pdists["cosine"]
    assert_row_close(mat, cosine_dist)
