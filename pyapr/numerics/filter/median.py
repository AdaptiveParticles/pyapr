from _pyaprwrapper.numerics.filter import *
import pyapr
from typing import Tuple

__allowed_sizes_median__ = [(x, x, x) for x in [3, 5, 7, 9, 11]] + [(1, x, x) for x in [3, 5, 7, 9, 11]]


def median_filter(apr: pyapr.APR,
                  parts: [pyapr.ShortParticles,
                          pyapr.FloatParticles],
                  size: Tuple[int, int, int] = (5, 5, 5)):

    if size not in __allowed_sizes_median__:
        raise ValueError(f'median_filter received an invalid argument \'size\' with value {size}. '
                         f'Allowed values are \n\t{__allowed_sizes_median__}')

    fname = 'median_filter_{}{}{}'.format(*size)
    output = pyapr.ShortParticles() if isinstance(parts, pyapr.ShortParticles) else pyapr.FloatParticles()
    globals()[fname](apr, parts, output)
    return output

