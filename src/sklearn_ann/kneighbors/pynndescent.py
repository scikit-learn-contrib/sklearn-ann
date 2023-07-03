from pynndescent import PyNNDescentTransformer as PyNNDescentTransformerBase


def no_op():
    pass


class PyNNDescentTransformer(PyNNDescentTransformerBase):
    def fit(self, X, compress_index=True):
        super().fit(X, compress_index=compress_index)
        self.index_.compress_index = no_op
        return self


__all__ = ["PyNNDescentTransformer"]
