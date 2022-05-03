from _pyaprwrapper.data_containers import APR, ByteParticles, ShortParticles, FloatParticles
from _pyaprwrapper.filter import local_std
from typing import Union, Optional, Tuple, List


ParticleData = Union[ByteParticles, ShortParticles, FloatParticles]


def std(apr: APR,
        parts: ParticleData,
        size: Union[int, Tuple[int, int, int], List[int]],
        output: Optional[FloatParticles]) -> FloatParticles:
    """
    Compute the local standard deviation in a neighborhood around each particle.

    Parameters
    ----------
    apr: APR
        Input APR data structure.
    parts: ParticleData
        Input particle values.
    size: int, tuple, list
        Size of the box in which standard deviations are computed. If a single integer is provided,
        considers a box of size ``min(size, apr.shape[dim])`` in each dimension. To use different sizes,
        give a list or tuple of length 3, specifying the size in dimensions (y, x, z)
    output: FloatParticles, optional
        (optional) Particle object to which the resulting values are written. If not provided, a new object
        is generated. (default: None)
    """
    if isinstance(size, int):
        size = (min(size, apr.org_dims(0)), min(size, apr.org_dims(1)), min(size, apr.org_dims(2)))
    else:
        assert isinstance(size, (tuple, list)), TypeError(f'argument \'size\' must be an integer, tuple or list, '
                                                          f'got {type(size)}')
        assert len(size) == 3, ValueError(f'argument size must be an integer or tuple/list of length 3, got {size}')
    output = output or FloatParticles()
    local_std(apr, parts, output, size)
    return output
