import numpy as np
from scipy.sparse.data import _data_matrix
from scipy.sparse import spmatrix, coo_matrix, sputils

from .base import _formats


class cic_matrix(_data_matrix):
    """
    Sparse CIrculant Column matrix
    Stores the first column as a sparse vector and caches a dense Fourier transform
    of the column, if required for multiplication.
    """
    
    format = 'cic'
    maxprint = 50
    
    def __init__(self, arg1, shape, dtype=None):
        super().__init__()
        
        self._shape = shape
        
        *arg1, self.shift = arg1
        
        self.shift %= self.shape[0]
        
        if len(arg1) == 1:
            data, = arg1
            if (
                isinstance(data, np.ndarray)
                and data.ndim == 1
                and len(data) == self.shape[0]
            ):
                # Dense column vector given
                self.offsets = data.nonzero()[0]
                self.data = data[self.offsets]
            else:
                raise ValueError("incorrect argument supplied")  # TODO: message
        else:
            self.data, self.offsets = arg1
        
        self.offsets %= self.shape[0]
        
        assert self.data.ndim == self.offsets.ndim == 1
        assert (
            len(self.data)
            == len(self.offsets)
            == len(np.unique(self.offsets))
            <= self.shape[0]
            > self.offsets.max()
            >= 0
        )
        
        self._fourier_column = None
        
        if dtype is not None:
            self.data = self.data.astype(dtype)
    
    def _with_data(self, data, copy=True):
        if copy:
            return cic_matrix((data, self.offsets.copy(), self.shift), shape=self.shape)
        else:
            return cic_matrix((data, self.offsets, self.shift), shape=self.shape)
    
    # TODO: slicing
    
    def __repr__(self):
        format = _formats[self.getformat()][1]
        return (
            f"<{self.shape[0]}x{self.shape[1]} sparse matrix of type"
            f" '{self.dtype.type}'\n\twith {self.nnz} stored elements in {format}"
            " format>"
        )
    
    def getnnz(self):
        return len(self.data)
    
    def count_nonzero(self):
        return np.count_nonzero(self.data)*self.shape[1]
    
    def tocoo(self, copy=False):
        v = np.tile(self.data, self.shape[1])
        i = np.concatenate(
            [
                (self.offsets + col*self.shift)%self.shape[0]
                for col in np.arange(self.shape[1])
            ]
        )
        j = np.repeat(np.arange(self.shape[1]), len(self.data))
        return coo_matrix((v, (i, j)), self.shape, dtype=self.dtype, copy=copy)
    
    tocoo.__doc__ = spmatrix.tocoo.__doc__
    
    def get_dense_column(self):
        col = np.zeros(self.shape[0], self.dtype)
        col[self.offsets] = self.data
        return col
    
    def get_fourier_column(self):
        if self._fourier_column is None:
            self._fourier_column = np.fft.rfft(self.get_dense_column())
        return self._fourier_column
    
    def tocir(self):
        if self.shape[0] != self.shape[1]:
            raise ValueError(
                "conversion from CIC to CIR is only supported for square matrices."
            )
        raise NotImplementedError("TODO")
    
    def transpose(self, axes=None, copy=False):
        # Avoiding circular imports
        from .cir import cir_matrix
        
        if axes is None:
            return cir_matrix(
                (self.data, self.offsets, self.shift), self.shape[::-1], self.dtype
            )
        else:
            return super().transpose(axes=axes, copy=copy)
    
    def sum(self, axis=None, dtype=None, out=None):
        if axis is None:
            return self.data.sum()*self.shape[1]
        elif axis == 0:
            return np.matrix(
                np.broadcast_to(self.data.sum(), (1, self.shape[1])), dtype=self.dtype
            ).sum(axis=(), dtype=dtype, out=out)
        else:
            return super().sum(axis=axis, dtype=dtype, out=out)
    
    def _is_compatible(self, other):
        return (
            isinstance(other, cic_matrix)
            and self.shape == other.shape
            and self.shift == other.shift
        )
    
    def multiply(self, other):
        if self._is_compatible(other):
            return cic_matrix(
                (self.get_dense_column()*other.get_dense_column(), self.shift),
                self.shape
            )
        else:
            return super().multiply(other)
    
    def maximum(self, other):
        if self._is_compatible(other):
            return cic_matrix(
                (
                    np.max((self.get_dense_column(), other.get_dense_column()), axis=0),
                    self.shift
                ),
                self.shape
            )
        else:
            return super().maximum(other)
    
    def minimum(self, other):
        if self._is_compatible(other):
            return cic_matrix(
                (
                    np.min((self.get_dense_column(), other.get_dense_column()), axis=0),
                    self.shift
                ),
                self.shape
            )
        else:
            return super().minimum(other)
    
    def _mul_vector(self, other):
        x = np.ravel(other)
        
        if self.shift == 0:
            # Very simple case, handle separately
            return self.get_dense_column()*x.sum()
        elif self.shift < 0:
            # Reverse input and roll first element back to first index
            x = np.concatenate(([x[0]], x[:0:-1]))
        
        shift = abs(self.shift)
        
        if shift == 1:
            x_folded = np.zeros(self.shape[0], dtype=other.dtype)
            for i in range(0, self.shape[1], self.shape[0]):
                x_part = x[i:i + self.shape[0]]
                x_folded[:len(x_part)] += x_part
        else:
            x_folded = np.bincount(
                (np.arange(len(x))*shift)%self.shape[0], x, self.shape[0]
            )
        
        return np.fft.irfft(
            self.get_fourier_column()*np.fft.rfft(x_folded), n=self.shape[0]
        ).astype(sputils.upcast_char(self.dtype.char, other.dtype.char))
    
    def _mul_multivector(self, other):
        return np.column_stack([self._mul_vector(column) for column in other.T])
