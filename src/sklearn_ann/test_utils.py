from __future__ import annotations

from enum import Enum
from importlib.util import find_spec
from typing import TYPE_CHECKING, overload

from scipy.sparse import csr_matrix

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import TypeVar

    import numpy as np
    import pytest
    from numpy.typing import NDArray

    Markable = TypeVar("Markable", bound=Callable[..., object] | type)


def assert_row_close(
    sp_mat: csr_matrix,
    actual_pdist: NDArray[np.float64],
    row: int = 42,
    thresh: float = 0.01,
) -> None:
    row_mat = sp_mat.getrow(row)
    assert isinstance(row_mat, csr_matrix)
    for col, val in zip(row_mat.indices, row_mat.data):
        assert abs(actual_pdist[row, col] - val) < thresh


class needs(Enum):
    """
    A pytest mark generator for skipping tests if a package is not installed.

    Can be used as a decorator:

    >>> @needs.faiss
    >>> def test_x(): pass

    or be called to create a mark object:

    >>> pytest.param(..., marks=[needs.annoy()])
    """

    annoy = ("annoy",)
    faiss = ("faiss-cpu", "faiss-gpu")
    nmslib = ("nmslib",)
    pynndescent = ("pynndescent",)

    @overload
    def __call__(self, fn: None = None) -> pytest.MarkDecorator: ...
    @overload
    def __call__(self, fn: Markable) -> Markable: ...
    def __call__(self, fn: Markable | None = None) -> Markable | pytest.MarkDecorator:
        import pytest

        what = (
            f"package {self.value[0]}"
            if len(self.value) == 1
            else f"one of the packages {set(self.value)}"
        )
        mark = pytest.mark.skipif(
            not find_spec(self.name),
            reason=f"`import {self.name}` needs {what} installed.",
        )
        return mark if fn is None else mark(fn)
