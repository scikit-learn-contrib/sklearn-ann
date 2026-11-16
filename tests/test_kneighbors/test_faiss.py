from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from sklearn_ann.test_utils import needs

if TYPE_CHECKING:
    from typing import Literal

    from numpy.typing import NDArray

try:
    from sklearn_ann.kneighbors.faiss import FAISSTransformer
except ImportError:
    pass


@needs.faiss
@pytest.mark.parametrize(
    ("metric", "scipy_metric"),
    [
        ("cosine", "cosine"),
        ("l1", "cityblock"),
        ("cityblock", "cityblock"),
        ("manhattan", "cityblock"),
        ("l2", "euclidean"),
        ("euclidean", "euclidean"),
        ("sqeuclidean", "sqeuclidean"),
        ("canberra", "canberra"),
        ("braycurtis", "braycurtis"),
        ("jensenshannon", "jensenshannon"),
    ],
)
def test_metrics(
    random_small: NDArray[np.float64],
    metric: str,
    scipy_metric: Literal[
        "cosine",
        "cityblock",
        "euclidean",
        "sqeuclidean",
        "canberra",
        "braycurtis",
        "jensenshannon",
    ],
) -> None:
    # probability vectors, so that jensenshannon is defined
    x = np.abs(random_small)
    x /= x.sum(axis=1, keepdims=True)
    pdists = cdist(x, x, scipy_metric)

    mat = FAISSTransformer(n_neighbors=5, metric=metric).fit_transform(x).tocoo()
    # correct distances, and the exact nearest neighbors
    np.testing.assert_allclose(mat.data, pdists[mat.row, mat.col], atol=1e-4)
    np.testing.assert_allclose(
        mat.data.reshape(len(x), 6), np.sort(pdists, axis=1)[:, :6], atol=1e-4
    )
