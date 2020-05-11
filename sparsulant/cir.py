import numpy as np
from scipy.sparse.data import _data_matrix
from scipy.sparse import spmatrix, coo_matrix, sputils

from .base import _formats


class cir_matrix(_data_matrix):
    """
    Sparse CIrculant Row matrix
    Stores the first row as a sparse vector.
    """
    
    format = 'cir'
    maxprint = 50
    
    def __init__(self, arg1, shape, dtype=None):
        super().__init__()
        
        self._shape = shape
        
        *arg1, self.shift = arg1
        
        self.shift %= self.shape[1]
        
        if len(arg1) == 1:
            data, = arg1
            if (
                isinstance(data, np.ndarray)
                and data.ndim == 1
                and len(data) == self.shape[1]
            ):
                # Dense row vector given
                self.offsets = data.nonzero()[0]
                self.data = data[self.offsets]
            else:
                raise ValueError("incorrect argument supplied")  # TODO: message
        else:
            self.data, self.offsets = arg1
        
        self.offsets %= self.shape[1]
        
        assert self.data.ndim == self.offsets.ndim == 1
        assert (
            len(self.data)
            == len(self.offsets)
            == len(np.unique(self.offsets))
            <= self.shape[1]
            > self.offsets.max()
            >= 0
        )
        
        self._conjugate_fourier_row = None
        
        if dtype is not None:
            self.data = self.data.astype(dtype)
    
    def _with_data(self, data, copy=True):
        if copy:
            return cir_matrix((data, self.offsets.copy(), self.shift), shape=self.shape)
        else:
            return cir_matrix((data, self.offsets, self.shift), shape=self.shape)
    
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
    
    @property
    def nbytes(self):
        return (
            self.data.nbytes
            + self.offsets.nbytes
            + (
                self._conjugate_fourier_row.nbytes
                if self._conjugate_fourier_row is not None
                else 0
            )
        )
    
    def tocoo(self, copy=False):
        v = np.tile(self.data, self.shape[0])
        i = np.repeat(np.arange(self.shape[0]), len(self.data))
        j = np.concatenate(
            [
                (self.offsets + row*self.shift)%self.shape[1]
                for row in np.arange(self.shape[0])
            ]
        )
        return coo_matrix((v, (i, j)), self.shape, dtype=self.dtype, copy=copy)
    
    tocoo.__doc__ = spmatrix.tocoo.__doc__
    
    def get_dense_row(self):
        col = np.zeros(self.shape[1], self.dtype)
        col[self.offsets] = self.data
        return col
    
    def get_conjugate_fourier_row(self):
        if self._conjugate_fourier_row is None:
            self._conjugate_fourier_row = np.fft.rfft(self.get_dense_row()).conj()
        return self._conjugate_fourier_row
    
    def tocic(self, copy=False):
        if self.shape[0] != self.shape[1]:
            raise ValueError(
                "conversion from CIR to CIC is only supported for square matrices."
            )
        raise NotImplementedError("TODO")
    
    def transpose(self, axes=None, copy=False):
        # Avoiding circular imports
        from .cic import cic_matrix
        
        if axes is None:
            return cic_matrix(
                (self.data, self.offsets, self.shift), self.shape[::-1], self.dtype
            )
        else:
            return super().transpose(axes=axes, copy=copy)
    
    def sum(self, axis=None, dtype=None, out=None):
        if axis is None:
            return self.data.sum()*self.shape[0]
        elif axis == 1:
            return np.matrix(
                np.broadcast_to(self.data.sum(), (self.shape[0], 1)), dtype=self.dtype
            ).sum(axis=(), dtype=dtype, out=out)
        else:
            return super().sum(axis=axis, dtype=dtype, out=out)
    
    def _is_compatible(self, other):
        return (
            isinstance(other, cir_matrix)
            and self.shape == other.shape
            and self.shift == other.shift
        )
    
    def multiply(self, other):
        if self._is_compatible(other):
            return cir_matrix(
                (self.get_dense_row()*other.get_dense_row(), self.shift), self.shape
            )
        else:
            return super().multiply(other)
    
    def maximum(self, other):
        if self._is_compatible(other):
            return cir_matrix(
                (
                    np.max((self.get_dense_row(), other.get_dense_row()), axis=0),
                    self.shift
                ),
                self.shape
            )
        else:
            return super().maximum(other)
    
    def minimum(self, other):
        if self._is_compatible(other):
            return cir_matrix(
                (
                    np.min((self.get_dense_row(), other.get_dense_row()), axis=0),
                    self.shift
                ),
                self.shape
            )
        else:
            return super().minimum(other)
    
    def _mul_vector(self, other):
        x = np.ravel(other)
        y = np.zeros(
            self.shape[0], dtype=sputils.upcast_char(self.dtype.char, x.dtype.char)
        )
        
        if self.shift == 0:
            y[:] = self.data.dot(x[self.offsets])
            return y
        
        period = min(self.shape[0], abs(np.lcm(self.shift, self.shape[1])//self.shift))
        
        y0 = np.fft.irfft(
            self.get_conjugate_fourier_row()*np.fft.rfft(x), n=self.shape[1]
        )
        
        y[:period] = y0[(self.shift*np.arange(period))%len(y0)]
        
        for i in range(period, self.shape[0], period):
            src = y[i - period:i]
            dst = y[i:i + period]
            size = min(len(src), len(dst))
            dst[:size] = src[:size]
        
        return y
    
    def _mul_multivector(self, other):
        y = np.zeros(
            (self.shape[0], other.shape[1]),
            dtype=sputils.upcast_char(self.dtype.char, other.dtype.char)
        )
        
        if self.shift == 0:
            y[:] = self.data.dot(other[self.offsets])
            return y
        
        period = min(self.shape[0], abs(np.lcm(self.shift, self.shape[1])//self.shift))
        
        y0 = np.fft.irfft(
            self.get_conjugate_fourier_row()[:, None]*np.fft.rfft(other, axis=0),
            n=self.shape[1],
            axis=0
        )
        
        y[:period] = y0[(self.shift*np.arange(period))%len(y0)]
        
        for i in range(period, self.shape[0], period):
            src = y[i - period:i]
            dst = y[i:i + period]
            size = min(len(src), len(dst))
            dst[:size] = src[:size]
        
        return y
