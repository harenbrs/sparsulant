from scipy.sparse import (
    bsr_matrix,
    coo_matrix,
    csc_matrix,
    csr_matrix,
    dia_matrix,
    dok_matrix,
    lil_matrix
)


def nbytes(mat):
    if hasattr(mat, 'nbytes'):
        return mat.nbytes
    elif isinstance(mat, (bsr_matrix, csr_matrix, csc_matrix)):
        return mat.data.nbytes + mat.indices.nbytes + mat.indptr.nbytes
    elif isinstance(mat, coo_matrix):
        return mat.data.nbytes + mat.row.nbytes + mat.col.nbytes
    elif isinstance(mat, dia_matrix):
        return mat.data.nbytes + mat.offsets.nbytes
    elif isinstance(mat, lil_matrix):
        return mat.data.nbytes + mat.rows.nbytes
    else:
        raise NotImplementedError
