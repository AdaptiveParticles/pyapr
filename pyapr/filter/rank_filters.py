from _pyaprwrapper.filter.rank_filters import *
from _pyaprwrapper.data_containers import APR, ByteParticles, ShortParticles, FloatParticles, LongParticles
from .._common import _check_input
from typing import Tuple, Union

ParticleData = Union[ByteParticles, ShortParticles, FloatParticles, LongParticles]

__allowed_sizes_median__ = [(x, x, x) for x in [3, 5, 7, 9, 11]] + [(1, x, x) for x in [3, 5, 7, 9, 11]]
__allowed_sizes_min__ = __allowed_sizes_median__
__allowed_sizes_max__ = __allowed_sizes_median__
__allowed_input_types__ = (ByteParticles, ShortParticles, FloatParticles, LongParticles)


def _check_size(size, allowed_sizes):
    if size not in allowed_sizes:
        raise ValueError(f'Invalid size {size}. Allowed values are {allowed_sizes}')


def median_filter(apr: APR,
                  parts: ParticleData,
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
    parts: ByteParticles, ShortParticles, FloatParticles or LongParticles
        Input particle values.
    size: (int, int, int)
        Size of the neighborhood in (z, x, y) dimensions.
        Allowed sizes are (x, x, x) and (1, x, x) for x in [3, 5, 7, 9, 11]. Default: (5, 5, 5)

    Returns
    -------
    output: ByteParticles, ShortParticles, FloatParticles or LongParticles
        Median filtered particle values of the same type as the input particles.
    """
    _check_input(apr, parts, __allowed_input_types__)
    _check_size(size, __allowed_sizes_median__)
    fname = 'median_filter_{}{}{}'.format(*size)
    output = type(parts)()
    globals()[fname](apr, parts, output)
    return output


def min_filter(apr: APR,
               parts: ParticleData,
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
    parts: ByteParticles, ShortParticles, FloatParticles or LongParticles
        Input particle values.
    size: (int, int, int)
        Size of the neighborhood in (z, x, y) dimensions.
        Allowed values are (x, x, x) and (1, x, x) for x in [3, 5, 7, 9, 11]. Default: (5, 5, 5)

    Returns
    -------
    output: ByteParticles, ShortParticles, FloatParticles or LongParticles
        Minimum filtered particle values of the same type as the input particles.
    """
    _check_input(apr, parts, __allowed_input_types__)
    _check_size(size, __allowed_sizes_min__)
    fname = 'min_filter_{}{}{}'.format(*size)
    output = type(parts)()
    globals()[fname](apr, parts, output)
    return output


def max_filter(apr: APR,
               parts: ParticleData,
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
    parts: ByteParticles, ShortParticles, FloatParticles or LongParticles
        Input particle values
    size: (int, int, int)
        Size of the neighborhood in (z, x, y) dimensions.
        Allowed values are (x, x, x) and (1, x, x) for x in [3, 5, 7, 9, 11]. Default: (5, 5, 5)

    Returns
    -------
    output: ByteParticles, ShortParticles, FloatParticles or LongParticles
        Maximum filtered particle values of the same type as the input particles.
    """
    _check_input(apr, parts, __allowed_input_types__)
    _check_size(size, __allowed_sizes_max__)
    fname = 'max_filter_{}{}{}'.format(*size)
    output = type(parts)()
    globals()[fname](apr, parts, output)
    return output
