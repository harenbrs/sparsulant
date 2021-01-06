from functools import lru_cache

import pytest
import numpy as np

from sparsulant import chb_matrix, cic_matrix, nbytes


@pytest.mark.benchmark(group='chb[cic]-vmul')
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
        
        return chb_matrix((cic, shift), shape), vector
    
    def test_chb_cic_vmul(
        self, n_blocks, block_shape, block_shift, shift, density, benchmark
    ):
        chb, vector = self.get_setup(n_blocks, block_shape, block_shift, shift, density)
        
        result = benchmark(chb._mul_vector, vector)
        
        assert np.allclose(result, chb.tocsr()._mul_vector(vector))
        
        benchmark.extra_info['memory'] = nbytes(chb)
    
    def test_chb_cic_vmul_baseline(
        self, n_blocks, block_shape, block_shift, shift, density, benchmark
    ):
        chb, vector = self.get_setup(n_blocks, block_shape, block_shift, shift, density)
        csr = chb.tocsr()
        
        benchmark(csr._mul_vector, vector)
        
        benchmark.extra_info['memory'] = nbytes(csr)


@pytest.mark.benchmark(group='chb[cic]-mmul')
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
        
        return chb_matrix((cic, shift), shape), matrix
    
    def test_chb_cic_mmul(
        self, n_blocks, block_shape, block_shift, shift, density, benchmark
    ):
        chb, matrix = self.get_setup(n_blocks, block_shape, block_shift, shift, density)
        
        result = benchmark(chb._mul_multivector, matrix)
        
        assert np.allclose(result, chb.tocsr()._mul_multivector(matrix))
        
        benchmark.extra_info['memory'] = nbytes(chb)
    
    def test_chb_cic_mmul_baseline(
        self, n_blocks, block_shape, block_shift, shift, density, benchmark
    ):
        chb, matrix = self.get_setup(n_blocks, block_shape, block_shift, shift, density)
        csr = chb.tocsr()
        
        benchmark(csr._mul_multivector, matrix)
        
        benchmark.extra_info['memory'] = nbytes(csr)


@pytest.mark.benchmark(group='chb[csr]-vmul')
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
        
        return chb_matrix((cic.tocsr(), shift), shape), vector
    
    def test_chb_csr_vmul(
        self, n_blocks, block_shape, block_shift, shift, density, benchmark
    ):
        chb, vector = self.get_setup(n_blocks, block_shape, block_shift, shift, density)
        
        result = benchmark(chb._mul_vector, vector)
        
        assert np.allclose(result, chb.tocsr()._mul_vector(vector))
        
        benchmark.extra_info['memory'] = nbytes(chb)
    
    def test_chb_csr_vmul_baseline(
        self, n_blocks, block_shape, block_shift, shift, density, benchmark
    ):
        chb, vector = self.get_setup(n_blocks, block_shape, block_shift, shift, density)
        csr = chb.tocsr()
        
        benchmark(csr._mul_vector, vector)
        
        benchmark.extra_info['memory'] = nbytes(csr)


@pytest.mark.benchmark(group='chb[csr]-mmul')
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
        
        return chb_matrix((cic.tocsr(), shift), shape), matrix
    
    def test_chb_csr_mmul(
        self, n_blocks, block_shape, block_shift, shift, density, benchmark
    ):
        chb, matrix = self.get_setup(n_blocks, block_shape, block_shift, shift, density)
        
        result = benchmark(chb._mul_multivector, matrix)
        
        assert np.allclose(result, chb.tocsr()._mul_multivector(matrix))
        
        benchmark.extra_info['memory'] = nbytes(chb)
    
    def test_chb_csr_mmul_baseline(
        self, n_blocks, block_shape, block_shift, shift, density, benchmark
    ):
        chb, matrix = self.get_setup(n_blocks, block_shape, block_shift, shift, density)
        csr = chb.tocsr()
        
        benchmark(csr._mul_multivector, matrix)
        
        benchmark.extra_info['memory'] = nbytes(csr)
