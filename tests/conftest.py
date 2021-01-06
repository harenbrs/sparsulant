from collections import defaultdict

import pytest
import numpy as np


shapes = {
    'tall': (10000, 1000),
    'square': (3162, 3162),
    'wide': (1000, 10000),
}

shifts = {
    'backward3': -3,
    'forward1': 1,
}

densities = {
    '1e-6': 1e-6,
    # '1e-4': 1e-4,
    # '1e-2': 1e-2,
    # '1.0': 1.0,
    'dense': 1
}

block_nums = {
    '1': 1,
    '3': 3,
}

block_shapes = {
    'tall-blocks': (3162, 316),
    'square-blocks': (1000, 1000),
    'wide-blocks': (316, 3162),
}

block_shifts = {
    '[backward3]': -3,
    '[forward1]': 1,
}


@pytest.fixture(ids=shapes.keys(), params=shapes.values())
def shape(request):
    return request.param


@pytest.fixture(ids=shifts.keys(), params=shifts.values())
def shift(request):
    return request.param


@pytest.fixture(ids=densities.keys(), params=densities.values())
def density(request):
    return request.param


@pytest.fixture(ids=block_nums.keys(), params=block_nums.values())
def n_blocks(request):
    return request.param


@pytest.fixture(ids=block_shapes.keys(), params=block_shapes.values())
def block_shape(request):
    return request.param


@pytest.fixture(ids=block_shifts.keys(), params=block_shifts.values())
def block_shift(request):
    return request.param
