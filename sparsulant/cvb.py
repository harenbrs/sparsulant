import numpy as np
import scipy.sparse
from scipy.sparse import spmatrix, coo_matrix, sputils

from .base import _formats
from .cic import cic_matrix
from .cir import cir_matrix
from .util import nbytes


class cvb_matrix(spmatrix):
    """
    Circulant Vertical Block matrix
    Stores the first block as a sparse matrix.
    """
    
    format = 'cvb'
    
    def __init__(self, arg1, shape, dtype=None):
        super().__init__()
        
        self._shape = shape
        
        self.block, self.shift = arg1
        
        # if not (isinstance(self.block, (cic_matrix, cir_matrix))):
        #     raise NotImplementedError("TODO")
        
        assert self.block.shape[1] == shape[1]
        assert shape[0]%self.block.shape[0] == 0
        
        self.n_blocks = self.shape[0]//self.block.shape[0]
        
        self.dtype = self.block.dtype
    
    # TODO: slicing
    
    def __repr__(self):
        format = _formats[self.getformat()][1]
        return (
            f"<{self.shape[0]}x{self.shape[1]} sparse matrix of type"
            f" '{self.dtype.type}'\n\twith {self.nnz} stored elements in {format}"
            " format>"
        )
    
    def getnnz(self):
        return self.block.getnnz()
    
    def count_nonzero(self):
        return self.block.count_nonzero()*self.n_blocks
    
    @property
    def nbytes(self):
        return nbytes(self.block)
    
    def transpose(self, axes=None, copy=False):
        from .chb import chb_matrix
        
        if axes is None:
            return chb_matrix((self.block.T, self.shift), self.shape[::-1], self.dtype)
        else:
            return super().transpose(axes=axes, copy=copy)
    
    def tocoo(self, copy=False):
        """
        Slow.
        """
        return scipy.sparse.vstack([self.get_block(i) for i in range(self.n_blocks)])
    
    def get_block(self, i=0):
        if i == 0:
            return self.block
        elif isinstance(self.block, cir_matrix):
            return cir_matrix(
                (self.block.data, self.block.offsets + i*self.shift, self.block.shift),
                self.block.shape,
                self.block.dtype
            )
        # elif isinstance(self.block, cir_matrix):
        #     raise NotImplementedError("TODO")
        else:
            coo = self.block.tocoo()
            return coo_matrix(
                (coo.data, (coo.row, (coo.col + i*self.shift)%coo.shape[1])),
                coo.shape,
                coo.dtype
            )
    
    def _mul_vector(self, other):
        x = np.ravel(other)
        y = np.zeros(
            self.shape[0], dtype=sputils.upcast_char(self.dtype.char, other.dtype.char)
        )
        
        if self.shift == 0:
            y0 = self.block @ x
            for i in range(self.n_blocks):
                y[i*len(y0):(i + 1)*len(y0)] = y0
            return y
        
        n0 = self.block.shape[0]
        period = min(self.n_blocks, abs(np.lcm(self.shift, self.shape[1])//self.shift))
        
        xr = np.empty_like(x)
        
        for i in range(period):
            # Equivalent to `xr = np.roll(x, -i*self.shift)``, but faster
            offset = -i*self.shift
            if offset == 0:
                xr[:] = x
            else:
                xr[:offset] = x[-offset:]
                xr[offset:] = x[:-offset]
            y[i*n0:(i + 1)*n0] += self.block @ xr
        
        row_period = n0*period
        y0 = y[:row_period]
        for i in range(row_period, self.shape[0], row_period):
            y[i:i + row_period] = y0[:len(y) - i]
        
        return y
    
    def _mul_multivector(self, other):
        y = np.zeros(
            (self.shape[0], other.shape[1]),
            dtype=sputils.upcast_char(self.dtype.char, other.dtype.char)
        )
        
        if self.shift == 0:
            y0 = self.block @ other
            for i in range(self.n_blocks):
                y[i*len(y0):(i + 1)*len(y0)] = y0
            return y
        
        n0 = self.block.shape[0]
        period = min(self.n_blocks, abs(np.lcm(self.shift, self.shape[1])//self.shift))
        
        xr = np.empty_like(other)
        
        for i in range(period):
            # Equivalent to `xr = np.roll(other, -i*self.shift, axis=0)`, but faster
            offset = -i*self.shift
            if offset == 0:
                xr[:] = other
            else:
                xr[:offset] = other[-offset:]
                xr[offset:] = other[:-offset]
            y[i*n0:(i + 1)*n0] += self.block @ xr
        
        row_period = n0*period
        y0 = y[:row_period]
        for i in range(row_period, self.shape[0], row_period):
            y[i:i + row_period] = y0[:len(y) - i]
        
        return y
