from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from numpy.random import default_rng
from scipy.spatial.distance import pdist, squareform

if TYPE_CHECKING:
    from typing import Literal

    import numpy as np
    from numpy.typing import NDArray


@pytest.fixture(scope="module")
def random_small() -> NDArray[np.float64]:
    gen = default_rng(42)
    return 2 * gen.random((64, 128)) - 1


@pytest.fixture(scope="module")
def random_small_pdists(
    random_small: NDArray[np.float64],
) -> dict[Literal["euclidean", "cosine"], NDArray[np.float64]]:
    metrics: list[Literal["euclidean", "cosine"]] = ["euclidean", "cosine"]
    return {
        metric: squareform(pdist(random_small, metric=metric)) for metric in metrics
    }
