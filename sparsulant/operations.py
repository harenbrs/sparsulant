import numpy as np

try:
    from numba import njit, prange
except ImportError:
    
    def njit(fn, *args, **kwargs):
        return fn
    
    prange = range


@njit(parallel=True)
def _cir_mul_vector(x, data, offsets, shift, period, shape, out):
    for i in prange(min(period, shape[0])):
        out[i] = data.dot(x[(offsets + i*shift)%shape[1]])
