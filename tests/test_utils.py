from __future__ import annotations

import numpy as np
import pytest

from sklearn_ann.utils import knn_csr


def test_knn_csr_drops_missing() -> None:
    ids = np.array([[1, 2], [0, -1], [-1, -1]])
    dist = np.array([[1.0, 2.0], [1.0, 9.0], [9.0, 9.0]])
    mat = knn_csr(ids, dist, 3)
    np.testing.assert_array_equal(mat.indptr, [0, 2, 3, 3])
    np.testing.assert_array_equal(mat.indices, [1, 2, 0])
    np.testing.assert_array_equal(mat.data, [1.0, 2.0, 1.0])


@pytest.mark.parametrize(
    ("negate", "sqrt", "raw", "expected"),
    [
        pytest.param(False, False, [-1e-7, 4.0], [0.0, 4.0], id="clip"),
        pytest.param(False, True, [-1e-7, 4.0], [0.0, 2.0], id="sqrt-no-nan"),
        pytest.param(True, False, [1 + 1e-7, 0.25], [0.0, 0.75], id="negate"),
    ],
)
def test_knn_csr_dist(
    *, negate: bool, sqrt: bool, raw: list[float], expected: list[float]
) -> None:
    mat = knn_csr(np.array([[0, 1]]), np.array([raw]), 2, negate=negate, sqrt=sqrt)
    np.testing.assert_array_equal(mat.data, expected)


def test_knn_csr_self_rows() -> None:
    ids = np.array(
        [
            [0, 1, 2],  # self first already
            [0, 1, 2],  # self not first (e.g. a tie at distance 0)
            [0, 1, 3],  # self missing: replaces the furthest
            [3, 3, 0],  # self repeated
            [0, -1, -1],  # self missing, fewer neighbors found
        ]
    )
    dist = np.array(
        [
            [1e-6, 1.0, 2.0],
            [0.0, 1e-6, 2.0],
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 1.0],
            [1.0, 9.0, 9.0],
        ]
    )
    mat = knn_csr(ids, dist, 5, self_rows=True)
    np.testing.assert_array_equal(mat.indptr, [0, 3, 6, 9, 11, 13])
    np.testing.assert_array_equal(mat.indices, [0, 1, 2, 1, 0, 2, 2, 0, 1, 3, 0, 4, 0])
    np.testing.assert_array_equal(
        mat.data, [0, 1, 2, 0, 0, 2, 0, 1, 2, 0, 1, 0, 1]
    )
