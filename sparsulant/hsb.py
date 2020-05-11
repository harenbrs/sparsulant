import numpy as np
import scipy.sparse
from scipy.sparse import spmatrix, sputils

from .base import _formats
from .util import nbytes


class hsb_matrix(spmatrix):
    """Horizontally Stacked Block matrix"""
    
    format = 'hsb'
    
    def __init__(self, blocks, dtype=None):
        ns, ms = zip(*[block.shape for block in blocks])
        assert min(ns) == max(ns)
        
        self._shape = ns[0], sum(ms)
        
        self.blocks = blocks
        
        self.dtype = dtype or np.dtype(
            sputils.upcast_char(*[block.dtype.char for block in blocks])
        )
    
    def __repr__(self):
        format = _formats[self.getformat()][1]
        return (
            f"<{self.shape[0]}x{self.shape[1]} sparse matrix of type"
            f" '{self.dtype.type}'\n\twith {self.nnz} stored elements in {format}"
            " format>"
        )
    
    def getnnz(self):
        return sum(block.getnnz() for block in self.blocks)
    
    def count_nonzero(self):
        return sum(block.count_nonzero() for block in self.blocks)
    
    @property
    def nbytes(self):
        return sum(map(nbytes, self.blocks))
    
    def transpose(self, axes=None, copy=False):
        from .vsb import vsb_matrix
        
        if axes is None:
            return vsb_matrix([block.T for block in self.blocks], dtype=self.dtype)
        else:
            super().transpose(axes=axes, copy=copy)
    
    def tocoo(self, copy=False):
        # TODO: efficiency?
        return scipy.sparse.hstack([block.tocoo() for block in self.blocks])
    
    def _mul_vector(self, other):
        x = np.ravel(other)
        
        y = np.zeros(
            self.shape[0], dtype=sputils.upcast_char(self.dtype.char, other.dtype.char)
        )
        
        offset = 0
        for block in self.blocks:
            y += block @ x[offset:offset + block.shape[1]]
            offset += block.shape[1]
        
        return y
    
    def _mul_multivector(self, other):
        y = np.zeros(
            (self.shape[0], other.shape[1]),
            dtype=sputils.upcast_char(self.dtype.char, other.dtype.char)
        )
        
        offset = 0
        for block in self.blocks:
            y += block @ other[offset:offset + block.shape[1]]
            offset += block.shape[1]
        
        return y
