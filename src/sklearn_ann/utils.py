import numpy as np
from scipy.sparse import csr_matrix


def check_metric(metric, metrics):
    if metric not in metrics:
        raise ValueError(f"Unknown metric {metric!r}. Valid metrics are {metrics!r}")


def get_sparse_indices(mat, idx):
    start_idx = mat.indptr[idx]
    end_idx = mat.indptr[idx + 1]
    return mat.indices[start_idx:end_idx]


def get_sparse_row(mat, idx):
    start_idx = mat.indptr[idx]
    end_idx = mat.indptr[idx + 1]
    return zip(mat.indices[start_idx:end_idx], mat.data[start_idx:end_idx])


def trunc_csr(csr, k):
    indptr = np.empty_like(csr.indptr)
    num_rows = len(csr.indptr) - 1
    indices = [None] * num_rows
    data = [None] * num_rows
    cur_indptr = 0
    for row_idx in range(num_rows):
        indptr[row_idx] = cur_indptr
        start_idx = csr.indptr[row_idx]
        old_end_idx = csr.indptr[row_idx + 1]
        end_idx = min(old_end_idx, start_idx + k)
        data[row_idx] = csr.data[start_idx:end_idx]
        indices[row_idx] = csr.indices[start_idx:end_idx]
        ptr_inc = min(k, old_end_idx - start_idx)
        cur_indptr = cur_indptr + ptr_inc
    indptr[-1] = cur_indptr
    return csr_matrix((np.concatenate(data), np.concatenate(indices), indptr))


def or_else_csrs(csr1, csr2):
    # Possible TODO: Could use numba/Cython to speed this up?
    if csr1.shape != csr2.shape:
        raise ValueError("csr1 and csr2 must be the same shape")
    indptr = np.empty_like(csr1.indptr)
    indices = []
    data = []
    for row_idx in range(len(indptr) - 1):
        indptr[row_idx] = len(indices)
        csr1_it = iter(get_sparse_row(csr1, row_idx))
        csr2_it = iter(get_sparse_row(csr2, row_idx))
        cur_csr1 = next(csr1_it, None)
        cur_csr2 = next(csr2_it, None)
        while 1:
            if cur_csr1 is None and cur_csr2 is None:
                break
            elif cur_csr1 is None:
                cur_index, cur_datum = cur_csr2
            elif cur_csr2 is None:
                cur_index, cur_datum = cur_csr1
            elif cur_csr1[0] < cur_csr2[0]:
                cur_index, cur_datum = cur_csr1
                cur_csr1 = next(csr1_it, None)
            elif cur_csr2[0] < cur_csr1[0]:
                cur_index, cur_datum = cur_csr2
                cur_csr2 = next(csr2_it, None)
            else:
                cur_index, cur_datum = cur_csr1
                cur_csr1 = next(csr1_it, None)
                cur_csr2 = next(csr2_it, None)
            indices.append(cur_index)
            data.append(cur_datum)
    indptr[-1] = len(indices)
    return csr_matrix((data, indices, indptr), shape=csr1.shape)


def postprocess_knn_csr(knns, include_fwd=True, include_rev=False):
    if not include_fwd and not include_rev:
        raise ValueError("One of include_fwd or include_rev must be True")
    elif include_rev and not include_fwd:
        return knns.transpose(copy=False)
    elif not include_rev and include_fwd:
        return knns
    else:
        inv_knns = knns.transpose(copy=True)
        return or_else_csrs(knns, inv_knns)


class TransformerChecksMixin:
    def _transform_checks(self, X, *fitted_props, **check_params):
        from sklearn.utils.validation import check_is_fitted

        X = self._validate_data(X, reset=False, **check_params)
        check_is_fitted(self, *fitted_props)
        return X
