from enum import Enum
from importlib.util import find_spec


def assert_row_close(sp_mat, actual_pdist, row=42, thresh=0.01):
    row_mat = sp_mat.getrow(row)
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

    def __call__(self, fn=None):
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
