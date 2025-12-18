from __future__ import annotations

from typing import TYPE_CHECKING

from pynndescent import PyNNDescentTransformer as PyNNDescentTransformerBase

if TYPE_CHECKING:
    from typing import Self

    from numpy.typing import ArrayLike


def no_op() -> None: ...


class PyNNDescentTransformer(PyNNDescentTransformerBase):
    def fit(self, X: ArrayLike, compress_index: bool = True) -> Self:
        super().fit(X, compress_index=compress_index)
        self.index_.compress_index = no_op
        return self


__all__ = ["PyNNDescentTransformer"]
