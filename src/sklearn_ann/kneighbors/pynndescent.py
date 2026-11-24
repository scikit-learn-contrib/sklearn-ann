from __future__ import annotations

from typing import TYPE_CHECKING

from pynndescent import PyNNDescentTransformer as PyNNDescentTransformerBase
from scipy.sparse import csr_array

from ..utils import knn_csr

if TYPE_CHECKING:
    from typing import Self

    from numpy.typing import ArrayLike


def no_op() -> None: ...


class PyNNDescentTransformer(PyNNDescentTransformerBase):
    def fit(self, X: ArrayLike, compress_index: bool = True) -> Self:
        super().fit(X, compress_index=compress_index)  # type: ignore[no-untyped-call]
        self.index_.compress_index = no_op  # type: ignore[method-assign]
        return self

    def transform(self, X: ArrayLike, y: None = None) -> csr_array:
        return csr_array(PyNNDescentTransformerBase.transform(self, X))

    def fit_transform(self, X: ArrayLike, y: None = None) -> csr_array:  # type: ignore[override]
        """Fit to `X` and return its kNN graph."""
        self.fit(X, compress_index=False)
        ids, dist = self.index_.neighbor_graph
        return knn_csr(
            ids,
            dist,
            self.n_samples_fit,
            self_rows=True,
            # -<x, y>: negative, and rows aren't their own nearest neighbors
            is_distance=self.metric != "inner_product",
        )


__all__ = ["PyNNDescentTransformer"]
