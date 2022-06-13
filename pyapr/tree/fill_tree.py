from _pyaprwrapper.tree import fill_tree_mean as _fill_tree_mean, \
                               fill_tree_min as _fill_tree_min, \
                               fill_tree_max as _fill_tree_max
from _pyaprwrapper.data_containers import APR, ByteParticles, ShortParticles, FloatParticles, LongParticles
from .._common import _check_input
from typing import Union, Optional


ParticleData = Union[ByteParticles, ShortParticles, FloatParticles, LongParticles]


def _check_output_type(parts: ParticleData, output: ParticleData) -> ParticleData:
    if output is None:
        return type(parts)()
    if not isinstance(output, (FloatParticles, type(parts))):
        raise TypeError(f'\'output\' must be None, FloatParticles or the same type as the input particles ({type(parts)})')
    return output


def fill_tree_mean(apr: APR,
                   parts: ParticleData,
                   output: Optional[ParticleData] = None) -> ParticleData:
    """
    Compute the values of all tree nodes (parent nodes of APR particles) by average downsampling.

    Parameters
    ----------
    apr: APR
        Input APR data structure.
    parts: ByteParticles, ShortParticles, FloatParticles or LongParticles
        Input APR particle values.
    output: ByteParticles, ShortParticles, FloatParticles or LongParticles, optional
        Particle data object for the output values. If provided, the type must either be FloatParticles
        or the same type as ``parts``. If None, generates a new object of the same type as ``parts``. (default: None)

    Returns
    -------
    output: ByteParticles, ShortParticles, FloatParticles or LongParticles
        The computed tree values.
    """
    _check_input(apr, parts)
    output = _check_output_type(parts, output)
    _fill_tree_mean(apr, parts, output)
    return output


def fill_tree_max(apr: APR,
                  parts: ParticleData,
                  output: Optional[ParticleData] = None) -> ParticleData:
    """
    Compute the values of all tree nodes (parent nodes of APR particles) by max downsampling.

    Parameters
    ----------
    apr: APR
        Input APR data structure.
    parts: ByteParticles, ShortParticles, FloatParticles or LongParticles
        Input APR particle values.
    output: ByteParticles, ShortParticles, FloatParticles or LongParticles, optional
        Particle data object for the output values. If provided, the type must either be FloatParticles
        or the same type as ``parts``. If None, generates a new object of the same type as ``parts``. (default: None)

    Returns
    -------
    output: ByteParticles, ShortParticles, FloatParticles or LongParticles
        The computed tree values.
    """
    _check_input(apr, parts)
    output = _check_output_type(parts, output)
    _fill_tree_max(apr, parts, output)
    return output


def fill_tree_min(apr: APR,
                  parts: ParticleData,
                  output: Optional[ParticleData] = None) -> ParticleData:
    """
    Compute the values of all tree nodes (parent nodes of APR particles) by min downsampling.

    Parameters
    ----------
    apr: APR
        Input APR data structure.
    parts: ByteParticles, ShortParticles, FloatParticles or LongParticles
        Input APR particle values.
    output: ByteParticles, ShortParticles, FloatParticles or LongParticles, optional
        Particle data object for the output values. If provided, the type must either be FloatParticles
        or the same type as ``parts``. If None, generates a new object of the same type as ``parts``. (default: None)

    Returns
    -------
    output: ByteParticles, ShortParticles, FloatParticles or LongParticles
        The computed tree values.
    """
    _check_input(apr, parts)
    output = _check_output_type(parts, output)
    _fill_tree_min(apr, parts, output)
    return output
