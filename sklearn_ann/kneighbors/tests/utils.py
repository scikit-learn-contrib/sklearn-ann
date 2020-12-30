def assert_row_close(sp_mat, actual_pdist, row=42, thresh=0.01):
    row_mat = sp_mat.getrow(row)
    for col, val in zip(row_mat.indices, row_mat.data):
        assert abs(actual_pdist[row, col] - val) < thresh
