from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from sklearn_ann.test_utils import assert_row_close, needs

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Literal

    from numpy.typing import NDArray

try:
    from sklearn_ann.kneighbors.mlxvs import MLXVSTransformer
except ImportError:
    pass


@needs.mlxvs
@pytest.mark.parametrize("algorithm", ["brute", "graph"])
@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_distances(
    random_small: NDArray[np.float64],
    random_small_pdists: Mapping[str, NDArray[np.float64]],
    algorithm: Literal["brute", "graph"],
    metric: str,
) -> None:
    trans = MLXVSTransformer(metric=metric, algorithm=algorithm)
    mat = trans.fit_transform(random_small)
    assert_row_close(mat, random_small_pdists[metric])
    mat = trans.transform(random_small)
    assert_row_close(mat, random_small_pdists[metric])


@needs.mlxvs
def test_graph_recall() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((20000, 16)).astype(np.float32)
    exact = MLXVSTransformer(n_neighbors=10, algorithm="brute").fit_transform(x)
    approx = MLXVSTransformer(n_neighbors=10, algorithm="graph").fit_transform(x)
    hits = [
        len(set(exact[i].indices) & set(approx[i].indices)) / 11
        for i in range(0, 20000, 7)
    ]
    assert np.mean(hits) > 0.95
    trans = MLXVSTransformer(n_neighbors=10, algorithm="graph").fit(x)
    queries = trans.transform(x[:500])
    # each fitted row is its own nearest neighbor
    assert all(queries[i].indices[np.argmin(queries[i].data)] == i for i in range(500))
