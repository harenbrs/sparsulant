from functools import lru_cache

import pytest
import numpy as np

from sparsulant import chb_matrix, cic_matrix, nbytes


@pytest.mark.benchmark(group='hsb[cic]-vmul')
class TestCICBlockVectorMultiplication:
    @lru_cache(maxsize=1, typed=True)
    def get_setup(self, n_blocks, block_shape, block_shift, shift, density):
        shape = (block_shape[0], n_blocks*block_shape[1])
        
        state = np.random.RandomState(0)
        column = state.uniform(-1, 1, shape[0])
        vector = state.uniform(-1, 1, shape[1])
        
        if isinstance(density, int) and density == 1:
            cic = cic_matrix((column, block_shift), block_shape)
        else:
            mask = state.uniform(0, 1, shape[0]) <= density
            data = column[mask]
            offsets, = np.nonzero(mask)
            cic = cic_matrix((data, offsets, block_shift), block_shape)
        
        return chb_matrix((cic, shift), shape).tohsb(), vector
    
    def test_hsb_cic_vmul(
        self, n_blocks, block_shape, block_shift, shift, density, benchmark
    ):
        hsb, vector = self.get_setup(n_blocks, block_shape, block_shift, shift, density)
        
        result = benchmark(hsb._mul_vector, vector)
        
        assert np.allclose(result, hsb.tocsr()._mul_vector(vector))
        
        benchmark.extra_info['memory'] = nbytes(hsb)
    
    def test_hsb_cic_vmul_baseline(
        self, n_blocks, block_shape, block_shift, shift, density, benchmark
    ):
        hsb, vector = self.get_setup(n_blocks, block_shape, block_shift, shift, density)
        csr = hsb.tocsr()
        
        benchmark(csr._mul_vector, vector)
        
        benchmark.extra_info['memory'] = nbytes(csr)


@pytest.mark.benchmark(group='hsb[cic]-mmul')
class TestCICBlockMatrixMultiplication:
    @lru_cache(maxsize=1, typed=True)
    def get_setup(self, n_blocks, block_shape, block_shift, shift, density):
        shape = (block_shape[0], n_blocks*block_shape[1])
        
        state = np.random.RandomState(0)
        column = state.uniform(-1, 1, shape[0])
        matrix = state.uniform(-1, 1, (shape[1], shape[1]//10))
        
        if isinstance(density, int) and density == 1:
            cic = cic_matrix((column, block_shift), block_shape)
        else:
            mask = state.uniform(0, 1, shape[0]) <= density
            data = column[mask]
            offsets, = np.nonzero(mask)
            cic = cic_matrix((data, offsets, block_shift), block_shape)
        
        return chb_matrix((cic, shift), shape).tohsb(), matrix
    
    def test_hsb_cic_mmul(
        self, n_blocks, block_shape, block_shift, shift, density, benchmark
    ):
        hsb, matrix = self.get_setup(n_blocks, block_shape, block_shift, shift, density)
        
        result = benchmark(hsb._mul_multivector, matrix)
        
        assert np.allclose(result, hsb.tocsr()._mul_multivector(matrix))
        
        benchmark.extra_info['memory'] = nbytes(hsb)
    
    def test_hsb_cic_mmul_baseline(
        self, n_blocks, block_shape, block_shift, shift, density, benchmark
    ):
        hsb, matrix = self.get_setup(n_blocks, block_shape, block_shift, shift, density)
        csr = hsb.tocsr()
        
        benchmark(csr._mul_multivector, matrix)
        
        benchmark.extra_info['memory'] = nbytes(csr)


@pytest.mark.benchmark(group='hsb[csr]-vmul')
class TestCSRBlockVectorMultiplication:
    @lru_cache(maxsize=1, typed=True)
    def get_setup(self, n_blocks, block_shape, block_shift, shift, density):
        shape = (block_shape[0], n_blocks*block_shape[1])
        
        state = np.random.RandomState(0)
        column = state.uniform(-1, 1, shape[0])
        vector = state.uniform(-1, 1, shape[1])
        
        if isinstance(density, int) and density == 1:
            cic = cic_matrix((column, block_shift), block_shape)
        else:
            mask = state.uniform(0, 1, shape[0]) <= density
            data = column[mask]
            offsets, = np.nonzero(mask)
            cic = cic_matrix((data, offsets, block_shift), block_shape)
        
        return chb_matrix((cic.tocsr(), shift), shape).tohsb(), vector
    
    def test_hsb_csr_vmul(
        self, n_blocks, block_shape, block_shift, shift, density, benchmark
    ):
        hsb, vector = self.get_setup(n_blocks, block_shape, block_shift, shift, density)
        
        result = benchmark(hsb._mul_vector, vector)
        
        assert np.allclose(result, hsb.tocsr()._mul_vector(vector))
        
        benchmark.extra_info['memory'] = nbytes(hsb)
    
    def test_hsb_csr_vmul_baseline(
        self, n_blocks, block_shape, block_shift, shift, density, benchmark
    ):
        hsb, vector = self.get_setup(n_blocks, block_shape, block_shift, shift, density)
        csr = hsb.tocsr()
        
        benchmark(csr._mul_vector, vector)
        
        benchmark.extra_info['memory'] = nbytes(csr)


@pytest.mark.benchmark(group='hsb[csr]-mmul')
class TestCSRBlockMatrixMultiplication:
    @lru_cache(maxsize=1, typed=True)
    def get_setup(self, n_blocks, block_shape, block_shift, shift, density):
        shape = (block_shape[0], n_blocks*block_shape[1])
        
        state = np.random.RandomState(0)
        column = state.uniform(-1, 1, shape[0])
        matrix = state.uniform(-1, 1, (shape[1], shape[1]//10))
        
        if isinstance(density, int) and density == 1:
            cic = cic_matrix((column, block_shift), block_shape)
        else:
            mask = state.uniform(0, 1, shape[0]) <= density
            data = column[mask]
            offsets, = np.nonzero(mask)
            cic = cic_matrix((data, offsets, block_shift), block_shape)
        
        return chb_matrix((cic.tocsr(), shift), shape).tohsb(), matrix
    
    def test_hsb_csr_mmul(
        self, n_blocks, block_shape, block_shift, shift, density, benchmark
    ):
        hsb, matrix = self.get_setup(n_blocks, block_shape, block_shift, shift, density)
        
        result = benchmark(hsb._mul_multivector, matrix)
        
        assert np.allclose(result, hsb.tocsr()._mul_multivector(matrix))
        
        benchmark.extra_info['memory'] = nbytes(hsb)
    
    def test_hsb_csr_mmul_baseline(
        self, n_blocks, block_shape, block_shift, shift, density, benchmark
    ):
        hsb, matrix = self.get_setup(n_blocks, block_shape, block_shift, shift, density)
        csr = hsb.tocsr()
        
        benchmark(csr._mul_multivector, matrix)
        
        benchmark.extra_info['memory'] = nbytes(csr)
