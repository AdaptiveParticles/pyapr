from _pyaprwrapper.numerics.filter import *
import pyapr
from typing import Tuple

__allowed_sizes_median__ = [(x, x, x) for x in [3, 5, 7, 9, 11]] + [(1, x, x) for x in [3, 5, 7, 9, 11]]
__allowed_sizes_min__ = [(x, x, x) for x in [3, 5]] + [(1, x, x) for x in [3, 5]]
__allowed_sizes_max__ = __allowed_sizes_min__


def median_filter(apr: pyapr.APR,
                  parts: [pyapr.ShortParticles,
                          pyapr.FloatParticles],
                  size: Tuple[int, int, int] = (5, 5, 5)):
    """

    """

    if size not in __allowed_sizes_median__:
        raise ValueError(f'median_filter received an invalid argument \'size\' = {size}. '
                         f'Allowed values are \n\t{__allowed_sizes_median__}')

    fname = 'median_filter_{}{}{}'.format(*size)
    output = pyapr.ShortParticles() if isinstance(parts, pyapr.ShortParticles) else pyapr.FloatParticles()
    globals()[fname](apr, parts, output)
    return output


def min_filter(apr: pyapr.APR,
               parts: [pyapr.ShortParticles,
                       pyapr.FloatParticles],
               size: Tuple[int, int, int] = (5, 5, 5)):

    if size not in __allowed_sizes_min__:
        raise ValueError(f'min_filter received an invalid argument \'size\' = {size}. '
                         f'Allowed values are \n\t{__allowed_sizes_min__}')

    fname = 'min_filter_{}{}{}'.format(*size)
    output = pyapr.ShortParticles() if isinstance(parts, pyapr.ShortParticles) else pyapr.FloatParticles()
    globals()[fname](apr, parts, output)
    return output


def max_filter(apr: pyapr.APR,
               parts: [pyapr.ShortParticles,
                       pyapr.FloatParticles],
               size: Tuple[int, int, int] = (5, 5, 5)):

    if size not in __allowed_sizes_max__:
        raise ValueError(f'max_filter received an invalid argument \'size\' = {size}. '
                         f'Allowed values are \n\t{__allowed_sizes_max__}')

    fname = 'max_filter_{}{}{}'.format(*size)
    output = pyapr.ShortParticles() if isinstance(parts, pyapr.ShortParticles) else pyapr.FloatParticles()
    globals()[fname](apr, parts, output)
    return output

