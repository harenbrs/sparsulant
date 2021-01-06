from functools import lru_cache

import pytest
import numpy as np

from sparsulant import cir_matrix, nbytes


@pytest.mark.benchmark(group='cir-vmul')
class TestVectorMultiplication:
    @lru_cache(maxsize=1, typed=True)
    def get_setup(self, shape, shift, density):
        state = np.random.RandomState(0)
        row = state.uniform(-1, 1, shape[1])
        vector = state.uniform(-1, 1, shape[1])
        
        if isinstance(density, int) and density == 1:
            return cir_matrix((row, shift), shape), vector
        else:
            mask = state.uniform(0, 1, shape[1]) <= density
            data = row[mask]
            offsets, = np.nonzero(mask)
            return cir_matrix((data, offsets, shift), shape), vector
    
    def test_cir_vmul(self, shape, shift, density, benchmark):
        cir, vector = self.get_setup(shape, shift, density)
        csr = cir.tocsr()
        
        result = benchmark(cir._mul_vector, vector)
        
        assert np.allclose(result, csr._mul_vector(vector))
        
        benchmark.extra_info['memory'] = nbytes(cir)
    
    def test_cir_vmul_baseline(self, shape, shift, density, benchmark):
        cir, vector = self.get_setup(shape, shift, density)
        csr = cir.tocsr()
        
        benchmark(csr._mul_vector, vector)
        
        benchmark.extra_info['memory'] = nbytes(csr)


@pytest.mark.benchmark(group='cir-mmul')
class TestMatrixMultiplication:
    @lru_cache(maxsize=1, typed=True)
    def get_setup(self, shape, shift, density):
        state = np.random.RandomState(0)
        row = state.uniform(-1, 1, shape[1])
        matrix = state.uniform(-1, 1, (shape[1], shape[1]//10))
        
        if isinstance(density, int) and density == 1:
            return cir_matrix((row, shift), shape), matrix
        else:
            mask = state.uniform(0, 1, shape[1]) <= density
            data = row[mask]
            offsets, = np.nonzero(mask)
            return cir_matrix((data, offsets, shift), shape), matrix
    
    def test_cir_mmul(self, shape, shift, density, benchmark):
        cir, matrix = self.get_setup(shape, shift, density)
        
        result = benchmark(cir._mul_multivector, matrix)
        
        assert np.allclose(result, cir.tocsr()._mul_multivector(matrix))
        
        benchmark.extra_info['memory'] = nbytes(cir)
    
    def test_cir_mmul_baseline(self, shape, shift, density, benchmark):
        cir, matrix = self.get_setup(shape, shift, density)
        csr = cir.tocsr()
        
        benchmark(csr._mul_multivector, matrix)
        
        benchmark.extra_info['memory'] = nbytes(csr)
