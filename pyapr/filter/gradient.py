from _pyaprwrapper.data_containers import APR, ByteParticles, ShortParticles, FloatParticles
from _pyaprwrapper.filter import gradient_cfd, gradient_sobel, gradient_magnitude_cfd, gradient_magnitude_sobel
from .._common import _check_input
from typing import Union, Optional, Tuple, List


ParticleData = Union[ByteParticles, ShortParticles, FloatParticles]
__allowed_input_types__ = (ByteParticles, ShortParticles, FloatParticles)


def gradient(apr: APR,
             parts: ParticleData,
             dim: int = 0,
             delta: float = 1.0,
             output: Optional[FloatParticles] = None) -> FloatParticles:
    """
    Compute the particle gradient in a given dimension using central finite differences. The resulting values are
    scaled based on the particle resolution level, to account for varying distances across the domain.

    Parameters
    ----------
    apr: APR
        Input APR data structure.
    parts: ParticleData
        Input particle values.
    dim: int
        Dimension (axis) along which the gradient is computed (0 -> y, 1 -> x, 2 -> z). (default: 0)
    delta: float
        Voxel size in the dimension of interest, used to scale the gradients. (default: 1.0)
    output: FloatParticles, optional
        (optional) Particle object to which the resulting values are written. If not provided, a new object
        is generated. (default: None)

    Returns
    -------
    output: FloatParticles
        The gradient value at each particle location.
    """
    _check_input(apr, parts, __allowed_input_types__)
    assert 0 <= dim <= 2, ValueError(f'argument \'dim\' must be an integer between 0 and 2, got {dim}.')
    output = output or FloatParticles()
    gradient_cfd(apr, parts, output, dim, delta)
    return output


def sobel(apr: APR,
          parts: ParticleData,
          dim: int = 0,
          delta: float = 1.0,
          output: Optional[FloatParticles] = None) -> FloatParticles:
    """
    Compute the particle gradient in a given dimension using the Sobel filter. The resulting values are
    scaled based on the particle resolution level, to account for varying distances across the domain.

    Parameters
    ----------
    apr: APR
        Input APR data structure.
    parts: ParticleData
        Input particle values.
    dim: int
        Dimension (axis) along which the gradient is computed (0 -> y, 1 -> x, 2 -> z). (default: 0)
    delta: float
        Voxel size in the dimension of interest, used to scale the gradients. (default: 1.0)
    output: FloatParticles, optional
        (optional) Particle object to which the resulting values are written. If not provided, a new object
        is generated. (default: None)

    Returns
    -------
    output: FloatParticles
        The gradient value at each particle location.
    """
    _check_input(apr, parts, __allowed_input_types__)
    assert 0 <= dim <= 2, ValueError(f'argument \'dim\' must be an integer between 0 and 2, got {dim}.')
    output = output or FloatParticles()
    gradient_sobel(apr, parts, output, dim, delta)
    return output


def gradient_magnitude(apr: APR,
                       parts: ParticleData,
                       deltas: Union[Tuple[float, float, float], List[float]] = (1.0, 1.0, 1.0),
                       output: Optional[FloatParticles] = None) -> FloatParticles:
    """
    Compute the particle gradient magnitude using central finite differences. The gradient values are
    scaled based on the particle resolution level, to account for varying distances across the domain.

    Parameters
    ----------
    apr: APR
        Input APR data structure.
    parts: ParticleData
        Input particle values.
    deltas: tuple or list of length 3
        Voxel size in dimensions (y, x, z) used to scale the gradients. (default: (1.0, 1.0, 1.0))
    output: FloatParticles, optional
        (optional) Particle object to which the resulting values are written. If not provided, a new object
        is generated. (default: None)

    Returns
    -------
    output: FloatParticles
        The gradient magnitude at each particle location.
    """
    _check_input(apr, parts, __allowed_input_types__)
    assert len(deltas) == 3, ValueError(f'argument \'deltas\' must be a tuple or list of length 3, got {deltas}.')
    output = output or FloatParticles()
    gradient_magnitude_cfd(apr, parts, output, deltas)
    return output


def sobel_magnitude(apr: APR,
                    parts: ParticleData,
                    deltas: Union[Tuple[float, float, float], List[float]] = (1.0, 1.0, 1.0),
                    output: Optional[FloatParticles] = None) -> FloatParticles:
    """
    Compute the particle gradient magnitude using Sobel filters. The gradient values are
    scaled based on the particle resolution level, to account for varying distances across the domain.

    Parameters
    ----------
    apr: APR
        Input APR data structure.
    parts: ParticleData
        Input particle values.
    deltas: tuple or list of length 3
        Voxel size in dimensions (y, x, z) used to scale the gradients. (default: (1.0, 1.0, 1.0))
    output: FloatParticles, optional
        (optional) Particle object to which the resulting values are written. If not provided, a new object
        is generated. (default: None)

    Returns
    -------
    output: FloatParticles
        The Sobel gradient magnitude at each particle location.
    """
    _check_input(apr, parts, __allowed_input_types__)
    assert len(deltas) == 3, ValueError(f'argument \'deltas\' must be a tuple or list of length 3, got {deltas}.')
    output = output or FloatParticles()
    gradient_magnitude_sobel(apr, parts, output, deltas)
    return output
