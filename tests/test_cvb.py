from functools import lru_cache

import pytest
import numpy as np

from sparsulant import cvb_matrix, cir_matrix, nbytes


@pytest.mark.benchmark(group='cvb[cir]-vmul')
class TestCIRBlockVectorMultiplication:
    @lru_cache(maxsize=1, typed=True)
    def get_setup(self, n_blocks, block_shape, block_shift, shift, density):
        shape = (n_blocks*block_shape[0], block_shape[1])
        
        state = np.random.RandomState(0)
        row = state.uniform(-1, 1, shape[1])
        vector = state.uniform(-1, 1, shape[1])
        
        if isinstance(density, int) and density == 1:
            cir = cir_matrix((row, block_shift), block_shape)
        else:
            mask = state.uniform(0, 1, shape[1]) <= density
            data = row[mask]
            offsets, = np.nonzero(mask)
            cir = cir_matrix((data, offsets, block_shift), block_shape)
        
        return cvb_matrix((cir, shift), shape), vector
    
    def test_cvb_cir_vmul(
        self, n_blocks, block_shape, block_shift, shift, density, benchmark
    ):
        cvb, vector = self.get_setup(n_blocks, block_shape, block_shift, shift, density)
        
        result = benchmark(cvb._mul_vector, vector)
        
        assert np.allclose(result, cvb.tocsr()._mul_vector(vector))
        
        benchmark.extra_info['memory'] = nbytes(cvb)
    
    def test_cvb_cir_vmul_baseline(
        self, n_blocks, block_shape, block_shift, shift, density, benchmark
    ):
        cvb, vector = self.get_setup(n_blocks, block_shape, block_shift, shift, density)
        csr = cvb.tocsr()
        
        benchmark(csr._mul_vector, vector)
        
        benchmark.extra_info['memory'] = nbytes(csr)


@pytest.mark.benchmark(group='cvb[cir]-mmul')
class TestCIRBlockMatrixMultiplication:
    @lru_cache(maxsize=1, typed=True)
    def get_setup(self, n_blocks, block_shape, block_shift, shift, density):
        shape = (n_blocks*block_shape[0], block_shape[1])
        
        state = np.random.RandomState(0)
        row = state.uniform(-1, 1, shape[1])
        matrix = state.uniform(-1, 1, (shape[1], shape[1]//10))
        
        if isinstance(density, int) and density == 1:
            cir = cir_matrix((row, block_shift), block_shape)
        else:
            mask = state.uniform(0, 1, shape[1]) <= density
            data = row[mask]
            offsets, = np.nonzero(mask)
            cir = cir_matrix((data, offsets, block_shift), block_shape)
        
        return cvb_matrix((cir, shift), shape), matrix
    
    def test_cvb_cir_mmul(
        self, n_blocks, block_shape, block_shift, shift, density, benchmark
    ):
        cvb, matrix = self.get_setup(n_blocks, block_shape, block_shift, shift, density)
        
        result = benchmark(cvb._mul_multivector, matrix)
        
        assert np.allclose(result, cvb.tocsr()._mul_multivector(matrix))
        
        benchmark.extra_info['memory'] = nbytes(cvb)
    
    def test_cvb_cir_mmul_baseline(
        self, n_blocks, block_shape, block_shift, shift, density, benchmark
    ):
        cvb, matrix = self.get_setup(n_blocks, block_shape, block_shift, shift, density)
        csr = cvb.tocsr()
        
        benchmark(csr._mul_multivector, matrix)
        
        benchmark.extra_info['memory'] = nbytes(csr)


@pytest.mark.benchmark(group='cvb[csr]-vmul')
class TestCSRBlockVectorMultiplication:
    @lru_cache(maxsize=1, typed=True)
    def get_setup(self, n_blocks, block_shape, block_shift, shift, density):
        shape = (n_blocks*block_shape[0], block_shape[1])
        
        state = np.random.RandomState(0)
        row = state.uniform(-1, 1, shape[1])
        vector = state.uniform(-1, 1, shape[1])
        
        if isinstance(density, int) and density == 1:
            cir = cir_matrix((row, block_shift), block_shape)
        else:
            mask = state.uniform(0, 1, shape[1]) <= density
            data = row[mask]
            offsets, = np.nonzero(mask)
            cir = cir_matrix((data, offsets, block_shift), block_shape)
        
        return cvb_matrix((cir.tocsr(), shift), shape), vector
    
    def test_cvb_csr_vmul(
        self, n_blocks, block_shape, block_shift, shift, density, benchmark
    ):
        cvb, vector = self.get_setup(n_blocks, block_shape, block_shift, shift, density)
        
        result = benchmark(cvb._mul_vector, vector)
        
        assert np.allclose(result, cvb.tocsr()._mul_vector(vector))
        
        benchmark.extra_info['memory'] = nbytes(cvb)
    
    def test_cvb_csr_vmul_baseline(
        self, n_blocks, block_shape, block_shift, shift, density, benchmark
    ):
        cvb, vector = self.get_setup(n_blocks, block_shape, block_shift, shift, density)
        csr = cvb.tocsr()
        
        benchmark(csr._mul_vector, vector)
        
        benchmark.extra_info['memory'] = nbytes(csr)


@pytest.mark.benchmark(group='cvb[csr]-mmul')
class TestCSRBlockMatrixMultiplication:
    @lru_cache(maxsize=1, typed=True)
    def get_setup(self, n_blocks, block_shape, block_shift, shift, density):
        shape = (n_blocks*block_shape[0], block_shape[1])
        
        state = np.random.RandomState(0)
        row = state.uniform(-1, 1, shape[1])
        matrix = state.uniform(-1, 1, (shape[1], shape[1]//10))
        
        if isinstance(density, int) and density == 1:
            cir = cir_matrix((row, block_shift), block_shape)
        else:
            mask = state.uniform(0, 1, shape[1]) <= density
            data = row[mask]
            offsets, = np.nonzero(mask)
            cir = cir_matrix((data, offsets, block_shift), block_shape)
        
        return cvb_matrix((cir.tocsr(), shift), shape), matrix
    
    def test_cvb_csr_mmul(
        self, n_blocks, block_shape, block_shift, shift, density, benchmark
    ):
        cvb, matrix = self.get_setup(n_blocks, block_shape, block_shift, shift, density)
        
        result = benchmark(cvb._mul_multivector, matrix)
        
        assert np.allclose(result, cvb.tocsr()._mul_multivector(matrix))
        
        benchmark.extra_info['memory'] = nbytes(cvb)
    
    def test_cvb_csr_mmul_baseline(
        self, n_blocks, block_shape, block_shift, shift, density, benchmark
    ):
        cvb, matrix = self.get_setup(n_blocks, block_shape, block_shift, shift, density)
        csr = cvb.tocsr()
        
        benchmark(csr._mul_multivector, matrix)
        
        benchmark.extra_info['memory'] = nbytes(csr)
