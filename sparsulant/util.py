import numpy as np
from scipy.sparse import (
    bsr_matrix,
    coo_matrix,
    csc_matrix,
    csr_matrix,
    dia_matrix,
    dok_matrix,
    lil_matrix,
    sputils,
    spmatrix,
    diags,
)

from .cic import cic_matrix
from .cir import cir_matrix
from .chb import chb_matrix
from .cvb import cvb_matrix


def nbytes(mat):
    if isinstance(mat, (np.matrix, np.ndarray)):
        return mat.nbytes
    elif isinstance(mat, (bsr_matrix, csr_matrix, csc_matrix)):
        return mat.data.nbytes + mat.indices.nbytes + mat.indptr.nbytes
    elif isinstance(mat, coo_matrix):
        return mat.data.nbytes + mat.row.nbytes + mat.col.nbytes
    elif isinstance(mat, (dia_matrix, cic_matrix, cir_matrix)):
        return mat.data.nbytes + mat.offsets.nbytes
    elif isinstance(mat, lil_matrix):
        return mat.data.nbytes + mat.rows.nbytes
    elif isinstance(mat, (chb_matrix, cvb_matrix)):
        return nbytes(mat.block)
    else:
        raise NotImplementedError
