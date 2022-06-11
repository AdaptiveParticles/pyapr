import pyapr
from .helpers import load_test_apr
import numpy as np
import math


def test_iterator_vs_slicer():
    apr, parts = load_test_apr(3)
    it = apr.iterator()

    for level_delta in [0, -1, -2]:
        slicer = pyapr.reconstruction.APRSlicer(apr, parts, level_delta=level_delta)
        level = it.level_max() + level_delta
        for z in range(5, 13):
            for x in range(1, 9):
                recon_row = slicer[z, x]
                for idx in range(it.begin(level, z, x), it.end()):
                    assert parts[idx] == recon_row[it.y(idx)]


def test_iterator_find_x():
    apr, parts = load_test_apr(3)
    it = apr.iterator()

    _shape = apr.shape()
    z_coords = [0] + list(np.random.randint(1, _shape[0]-1, size=4)) + [_shape[0]-1]

    for z in z_coords:
        for x in range(_shape[1]):
            for y in range(_shape[2]):
                # find particle at z, x, y
                idx = it.find_particle(z, x, y)

                # find coordinates of particle
                level, z_l, x_l, y_l = it.find_coordinates(idx)
                size_factor = 2 ** (it.level_max() - level)
                assert z_l == (z // size_factor)
                assert x_l == (x // size_factor)
                assert y_l == (y // size_factor)
