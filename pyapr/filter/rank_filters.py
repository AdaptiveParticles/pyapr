from _pyaprwrapper.filter.rank_filters import *
from _pyaprwrapper.data_containers import APR, ShortParticles, FloatParticles
from .._common import _check_input
from typing import Tuple, Union

__allowed_sizes_median__ = [(x, x, x) for x in [3, 5, 7, 9, 11]] + [(1, x, x) for x in [3, 5, 7, 9, 11]]
__allowed_sizes_min__ = [(x, x, x) for x in [3, 5]] + [(1, x, x) for x in [3, 5]]
__allowed_sizes_max__ = __allowed_sizes_min__
__allowed_input_types__ = (ShortParticles, FloatParticles)


def _check_size(size, allowed_sizes):
    assert size in allowed_sizes, ValueError(f'Invalid size {size}. Allowed values are {allowed_sizes}')


def median_filter(apr: APR,
                  parts: Union[ShortParticles, FloatParticles],
                  size: Tuple[int, int, int] = (5, 5, 5)):
    """
    Apply median filter to an APR image and return a new set of particle values.
    Each output particle is the median of an isotropic neighborhood of the given size
    in the input image. At coarse resolutions, neighboring values at finer resolution
    are average downsampled.

    Parameters
    ----------
    apr: APR
        APR data structure.
    parts: ShortParticles or FloatParticles
        Input particle values.
    size: (int, int, int)
        Size of the neighborhood in (z, x, y) dimensions.
        Allowed sizes are (x, x, x) and (1, x, x) for x in [3, 5, 7, 9, 11].

    Returns
    -------
    output: ShortParticles or FloatParticles
        Median filtered particle values of the same type as the input.
    """
    _check_input(apr, parts, __allowed_input_types__)
    _check_size(size, __allowed_sizes_median__)
    fname = 'median_filter_{}{}{}'.format(*size)
    output = ShortParticles() if isinstance(parts, ShortParticles) else FloatParticles()
    globals()[fname](apr, parts, output)
    return output


def min_filter(apr: APR,
               parts: Union[ShortParticles, FloatParticles],
               size: Tuple[int, int, int] = (5, 5, 5)):
    """
    Apply minimum filter to an APR image and return a new set of particle values.
    Each output particle is the minimum of an isotropic neighborhood of the given size
    in the input image. At coarse resolutions, neighboring values at finer resolution
    are minimum downsampled.

    Parameters
    ----------
    apr: APR
        APR data structure.
    parts: ShortParticles or FloatParticles
        Input particle values.
    size: (int, int, int)
        Size of the neighborhood in (z, x, y) dimensions.
        Allowed values are (x, x, x) and (1, x, x) for x in [3, 5].

    Returns
    -------
    output: ShortParticles or FloatParticles
        Minimum filtered particle values of the same type as the input.
    """
    _check_input(apr, parts, __allowed_input_types__)
    _check_size(size, __allowed_sizes_min__)
    fname = 'min_filter_{}{}{}'.format(*size)
    output = ShortParticles() if isinstance(parts, ShortParticles) else FloatParticles()
    globals()[fname](apr, parts, output)
    return output


def max_filter(apr: APR,
               parts: Union[ShortParticles, FloatParticles],
               size: Tuple[int, int, int] = (5, 5, 5)):
    """
    Apply maximum filter to an APR image and return a new set of particle values.
    Each output particle is the minimum of an isotropic neighborhood of the given size
    in the input image. At coarse resolutions, neighboring values at finer resolution
    are maximum downsampled.

    Parameters
    ----------
    apr: APR
        APR data structure.
    parts: ShortParticles or FloatParticles
        Input particle values
    size: (int, int, int)
        Size of the neighborhood in (z, x, y) dimensions.
        Allowed values are (x, x, x) and (1, x, x) for x in [3, 5].

    Returns
    -------
    output: ShortParticles or FloatParticles
        Maximum filtered particle values of the same type as the input.
    """
    _check_input(apr, parts, __allowed_input_types__)
    _check_size(size, __allowed_sizes_max__)
    fname = 'max_filter_{}{}{}'.format(*size)
    output = ShortParticles() if isinstance(parts, ShortParticles) else FloatParticles()
    globals()[fname](apr, parts, output)
    return output

