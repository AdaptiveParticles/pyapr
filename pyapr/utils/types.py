from _pyaprwrapper.data_containers import ByteParticles, ShortParticles, FloatParticles, LongParticles, \
                                          LazyDataByte, LazyDataShort, LazyDataFloat, LazyDataLong
import numpy as np
from typing import Union


ParticleData = Union[ByteParticles, ShortParticles, LongParticles, FloatParticles]
LazyData = Union[LazyDataByte, LazyDataShort, LazyDataLong, LazyDataFloat]


def type_to_particles(typespec: Union[str, type]) -> ParticleData:
    """
    Returns a ParticleData object of the specified type.

    Parameters
    ----------
    typespec: str or type
        String specifying the data type. Valid types are ``uint8``, ``uint16``, ``uint64``, ``float`` or corresponding
        numpy types.

    Returns
    -------
    parts: ByteParticles, ShortParticles, FloatParticles or LongParticles
        ParticleData of the specified type (if valid).
    """

    if typespec in ('uint16', np.uint16):
        return ShortParticles()
    if typespec in ('float', 'float32', np.float32):
        return FloatParticles()
    if typespec in ('uint8', np.uint8):
        return ByteParticles()
    if typespec in ('uint64', np.uint64):
        return LongParticles()
    raise ValueError(f'Type {typespec} is currently not supported. Valid types are \'uint8\', \'uint16\', '
                     f'\'uint64\' and \'float\'')


def type_to_lazy_particles(typespec: Union[str, type]) -> LazyData:
    """
    Returns a LazyData object of the specified type.

    Parameters
    ----------
    typespec: str or type
        String specifying the data type. Valid types are ``uint8``, ``uint16``, ``uint64``, ``float`` or corresponding
        numpy types.

    Returns
    -------
    parts: LazyDataByte, LazyDataShort, LazyDataLong or LazyDataFloat
        LazyData of the specified type (if valid).
    """

    if typespec in ('uint16', np.uint16):
        return LazyDataShort()
    if typespec in ('float', 'float32', np.float32):
        return LazyDataFloat()
    if typespec in ('uint8', np.uint8):
        return LazyDataByte()
    if typespec in ('uint64', np.uint64):
        return LazyDataLong()
    raise ValueError(f'Type {typespec} is currently not supported. Valid types are \'uint8\', \'uint16\', '
                     f'\'uint64\' and \'float\' (\'float32\')')


def particles_to_type(parts: Union[ParticleData, LazyData]) -> type:
    """
    Returns the numpy dtype corresponding to a ParticleData or LazyData object.

    Parameters
    ----------
    parts: ByteParticles, ShortParticles, LongParticles, FloatParticles, LazyDataByte, LazyDataShort, LazyDataLong or LazyDataFloat
        Particle data object.

    Returns
    -------
    output: type
        numpy type corresponding to the data type of the input object.
    """

    if isinstance(parts, (ShortParticles, LazyDataShort)):
        return np.uint16
    if isinstance(parts, (FloatParticles, LazyDataFloat)):
        return np.float32
    if isinstance(parts, (ByteParticles, LazyDataByte)):
        return np.uint8
    if isinstance(parts, (LongParticles, LazyDataLong)):
        return np.uint64
    raise TypeError(f'Input must be of type {ParticleData} or {LazyData} ({type(parts)} was provided)')
