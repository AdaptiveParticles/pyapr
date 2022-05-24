from _pyaprwrapper.data_containers import APR, ShortParticles, FloatParticles
from _pyaprwrapper.filter import convolve as _convolve, convolve_pencil as _convolve_pencil
from _pyaprwrapper import __cuda_build__
try:
    from _pyaprwrapper.filter import convolve_cuda as _convolve_cuda
except ImportError:
    _convolve_cuda = _convolve_pencil
from .._common import _check_input
import numpy as np
from warnings import warn
from typing import Union, Optional


__allowed_input_types__ = (ShortParticles, FloatParticles)
ParticleData = Union[ShortParticles, FloatParticles]


def __check_stencil(stencil: np.ndarray):
    """Expand stencil to 3 dimensions and convert it to float32 type"""
    while stencil.ndim < 3:
        stencil = np.expand_dims(stencil, axis=0)
    return stencil.astype(np.float32)


def __check_method(method: str, stencil: np.ndarray):
    if method == 'cuda':
        if not __cuda_build__:
            warn(f'Method \'cuda\' requires pyapr to be built with CUDA support (see installation instructions), '
                 f'using method \'pencil\' on CPU.', UserWarning)
            method = 'pencil'

        if stencil.shape not in [(3, 3, 3), (5, 5, 5)]:
            warn(f'Method \'cuda\' currently only supports stencils of shape (3, 3, 3) and (5, 5, 5), '
                 f'but got {stencil.shape}. Using method \'pencil\' on CPU.', UserWarning)
            method = 'pencil'
    return method


def correlate(apr: APR,
              parts: ParticleData,
              stencil: np.ndarray,
              output: Optional[FloatParticles] = None,
              restrict_stencil: bool = True,
              normalize_stencil: bool = True,
              reflect_boundary: bool = True,
              method: str = 'pencil') -> FloatParticles:
    """
    Compute the spatial cross-correlation between an APR image and a stencil. The output value at each particle
    location is computed by interpolating neighboring particle values to the resolution of the target particle
    and applying the stencil.

    The operation does not modify the sampling of the input APR.

    Parameters
    ----------
    apr: APR
        Input APR object
    parts: ShortParticles, FloatParticles
        Input particle values
    stencil: np.ndarray
        Stencil or kernel to correlate with the image. Should be 3-dimensional and of type float32, otherwise
        it is expanded, e.g. shape (3, 3) -> (1, 3, 3), and cast.
    output: FloatParticles, optional
        (optional) Particle object to which the resulting values are written. If not provided, a new object
        is generated (default: None)
    restrict_stencil: bool
        If True, the stencil is adapted to coarser resolution levels such that the correlation is consistent with
        applying ``stencil`` to the reconstructed pixel image. (default: True)
    normalize_stencil: bool
        If True, the stencil is normalized to sum to 1 (if ``restrict_stencil`` is True, the stencil is normalized
        at each resolution level. (default: True)
    reflect_boundary: bool
        If True, values are reflected at the boundary. Otherwise, zero padding is used. (default: True)
    method: str
        Method used to apply the operation:

            - 'pencil': construct isotropic neighborhoods of shape (stencil.shape[0], stencil.shape[1], apr.shape[2])
            - 'slice': construct isotropic neighborhoods of shape (stencil.shape[0], apr.shape[1], apr.shape[2])
            - 'cuda': compute the correlation using the GPU. Requires the package to be built with CUDA support,
              and ``stencil`` to have shape (3, 3, 3) or (5, 5, 5).

        The methods may differ in performance, depending on the input data, but produce the same result. (default: 'pencil')

    Returns
    -------
    output: FloatParticles
        The cross-correlation value at each particle location.
    """

    _check_input(apr, parts, __allowed_input_types__)
    stencil = __check_stencil(stencil)
    method = __check_method(method, stencil)
    output = output or FloatParticles()

    if method == 'pencil':
        _convolve_pencil(apr, parts, output, stencil, restrict_stencil, normalize_stencil, reflect_boundary)
    elif method == 'slice':
        _convolve(apr, parts, output, stencil, restrict_stencil, normalize_stencil, reflect_boundary)
    elif method == 'cuda':
        _convolve_cuda(apr, parts, output, stencil, restrict_stencil, normalize_stencil, reflect_boundary)
    else:
        raise ValueError(f'method {method} not recognized. Allowed values are \'pencil\', \'slice\' and \'cuda\'.')
    return output


def convolve(apr: APR,
             parts: ParticleData,
             stencil: np.ndarray,
             output: Optional[FloatParticles] = None,
             restrict_stencil: bool = True,
             normalize_stencil: bool = True,
             reflect_boundary: bool = True,
             method: str = 'pencil') -> FloatParticles:
    """
    Compute the spatial convolution between an APR image and a stencil. The output value at each particle
    location is computed by interpolating neighboring particle values to the resolution of the target particle
    and applying the stencil.

    The operation does not modify the sampling of the input APR.

    Parameters
    ----------
    apr: APR
        Input APR object
    parts: ShortParticles, FloatParticles
        Input particle values
    stencil: np.ndarray
        Stencil or kernel to convolve with the image. Should be 3-dimensional and of type float32, otherwise
        it is expanded, e.g. shape (3, 3) -> (1, 3, 3), and cast.
    output: FloatParticles, optional
        (optional) Particle object to which the resulting values are written. If not provided, a new object
        is generated (default: None)
    restrict_stencil: bool
        If True, the stencil is adapted to coarser resolution levels such that the convolution is consistent with
        applying ``stencil`` to the reconstructed pixel image. (default: True)
    normalize_stencil: bool
        If True, the stencil is normalized to sum to 1 (if ``restrict_stencil`` is True, the stencil is normalized
        at each resolution level. (default: True)
    reflect_boundary: bool
        If True, values are reflected at the boundary. Otherwise, zero padding is used. (default: True)
    method: str
        Method used to apply the operation:

            - 'pencil': construct isotropic neighborhoods of shape (stencil.shape[0], stencil.shape[1], apr.shape[2])
            - 'slice': construct isotropic neighborhoods of shape (stencil.shape[0], apr.shape[1], apr.shape[2])
            - 'cuda': compute the convolution using the GPU. Requires the package to be built with CUDA support,
              and ``stencil`` to have shape (3, 3, 3) or (5, 5, 5).

        The methods may differ in performance, depending on the input data, but produce the same result. (default: 'pencil')

    Returns
    -------
    output: FloatParticles
        The convolution value at each particle location.
    """

    _check_input(apr, parts, __allowed_input_types__)
    stencil = np.ascontiguousarray(np.flip(__check_stencil(stencil)))
    method = __check_method(method, stencil)
    output = output or FloatParticles()

    if method == 'pencil':
        _convolve_pencil(apr, parts, output, stencil, restrict_stencil, normalize_stencil, reflect_boundary)
    elif method == 'slice':
        _convolve(apr, parts, output, stencil, restrict_stencil, normalize_stencil, reflect_boundary)
    elif method == 'cuda':
        _convolve_cuda(apr, parts, output, stencil, restrict_stencil, normalize_stencil, reflect_boundary)
    else:
        raise ValueError(f'method {method} not recognized. Allowed values are \'pencil\', \'slice\' and \'cuda\'.')
    return output
