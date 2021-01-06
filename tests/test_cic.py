from functools import lru_cache

import pytest
import numpy as np

from sparsulant import cic_matrix, nbytes


@pytest.mark.benchmark(group='cic-vmul')
class TestVectorMultiplication:
    @lru_cache(maxsize=1, typed=True)
    def get_setup(self, shape, shift, density):
        state = np.random.RandomState(0)
        column = state.uniform(-1, 1, shape[0])
        vector = state.uniform(-1, 1, shape[1])
        
        if isinstance(density, int) and density == 1:
            return cic_matrix((column, shift), shape), vector
        else:
            mask = state.uniform(0, 1, shape[0]) <= density
            data = column[mask]
            offsets, = np.nonzero(mask)
            return cic_matrix((data, offsets, shift), shape), vector
    
    def test_cic_vmul(self, shape, shift, density, benchmark):
        cic, vector = self.get_setup(shape, shift, density)
        
        result = benchmark(cic._mul_vector, vector)
        
        assert np.allclose(result, cic.tocsr()._mul_vector(vector))
        
        benchmark.extra_info['memory'] = nbytes(cic)
    
    def test_cic_vmul_baseline(self, shape, shift, density, benchmark):
        cic, vector = self.get_setup(shape, shift, density)
        csr = cic.tocsr()
        
        benchmark(csr._mul_vector, vector)
        
        benchmark.extra_info['memory'] = nbytes(csr)


@pytest.mark.benchmark(group='cic-mmul')
class TestMatrixMultiplication:
    @lru_cache(maxsize=1, typed=True)
    def get_setup(self, shape, shift, density):
        state = np.random.RandomState(0)
        column = state.uniform(-1, 1, shape[0])
        matrix = state.uniform(-1, 1, (shape[1], shape[1]//10))
        
        if isinstance(density, int) and density == 1:
            return cic_matrix((column, shift), shape), matrix
        else:
            mask = state.uniform(0, 1, shape[0]) <= density
            data = column[mask]
            offsets, = np.nonzero(mask)
            return cic_matrix((data, offsets, shift), shape), matrix
    
    def test_cic_mmul(self, shape, shift, density, benchmark):
        cic, matrix = self.get_setup(shape, shift, density)
        
        result = benchmark(cic._mul_multivector, matrix)
        
        assert np.allclose(result, cic.tocsr()._mul_multivector(matrix))
        
        benchmark.extra_info['memory'] = nbytes(cic)
    
    def test_cic_mmul_baseline(self, shape, shift, density, benchmark):
        cic, matrix = self.get_setup(shape, shift, density)
        csr = cic.tocsr()
        
        benchmark(csr._mul_multivector, matrix)
        
        benchmark.extra_info['memory'] = nbytes(csr)
