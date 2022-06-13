from _pyaprwrapper.data_containers import APR, ByteParticles, ShortParticles, FloatParticles
import _pyaprwrapper.restoration as _internals
from _pyaprwrapper import __cuda_build__
from ..filter.convolution import __check_stencil
from .._common import _check_input
import numpy as np
from warnings import warn
from typing import Union, Optional

__allowed_input_types__ = (ByteParticles, ShortParticles, FloatParticles)
ParticleData = Union[ByteParticles, ShortParticles, FloatParticles]


def _check_output(apr: APR, output: Optional[FloatParticles]):
    if not isinstance(output, FloatParticles) or len(output) != apr.total_number_particles():
        raise ValueError(f'Option \'resume\' enabled but \'output\' is not properly initialized. Expected '
                         f'FloatParticles(size {apr.total_number_particles()}) but got {output}.')


def richardson_lucy(apr: APR,
                    parts: ParticleData,
                    psf: np.ndarray,
                    num_iter: int = 10,
                    restrict_psf: bool = True,
                    normalize_psf: bool = True,
                    resume: bool = False,
                    output: Optional[FloatParticles] = None) -> FloatParticles:
    """
    Richardson-Lucy deconvolution

    Parameters
    ----------
    apr: APR
        Input APR object.
    parts: ByteParticles, ShortParticles or FloatParticles
        Input particle values.
    psf: np.ndarray
        Point spread function. Should be 3-dimensional and of type float32, otherwise
        it is expanded, e.g. shape (3, 3) -> (1, 3, 3), and cast.
    num_iter: int
        Number of iterations. (default: 10)
    restrict_psf: bool
        If True, the psf is adapted to coarser resolution levels such that the forward blur is consistent with
        applying the psf to the reconstructed pixel image. (default: True)
    normalize_psf: bool
        If True, the psf is normalized to sum to 1 (if ``restrict_psf`` is True, it is normalized
        at each resolution level. (default: True)
    resume: bool
        Resume iterations from a previous estimate? If True, the previous estimate must be provided
        via the ``output`` argument. (default: False)
    output: FloatParticles, optional
        Particle object to which the resulting values are written, or the initial estimate if
        ``resume`` is enabled. If not provided, a new object is generated (default: None)

    Returns
    -------
    output: FloatParticles
        The restored particle intensities.
    """

    _check_input(apr, parts, __allowed_input_types__)
    psf = __check_stencil(psf)
    if resume:
        _check_output(apr, output)
    output = output if isinstance(output, FloatParticles) else FloatParticles()
    _internals.richardson_lucy(apr, parts, output, psf, num_iter, restrict_psf, normalize_psf, resume)
    return output


def richardson_lucy_tv(apr: APR,
                       parts: ParticleData,
                       psf: np.ndarray,
                       reg_param: float = 1e-2,
                       num_iter: int = 10,
                       restrict_psf: bool = True,
                       normalize_psf: bool = True,
                       resume: bool = False,
                       output: Optional[FloatParticles] = None) -> FloatParticles:
    """
    Richardson-Lucy deconvolution with total variation regularization.

    Parameters
    ----------
    apr: APR
        Input APR object.
    parts: ByteParticles, ShortParticles or FloatParticles
        Input particle values.
    psf: np.ndarray
        Point spread function. Should be 3-dimensional and of type float32, otherwise
        it is expanded, e.g. shape (3, 3) -> (1, 3, 3), and cast.
    reg_param: float
        Regularization parameter controlling the weight of the TV regularization term. (default: 1e-2)
    num_iter: int
        Number of iterations. (default: 10)
    restrict_psf: bool
        If True, the psf is adapted to coarser resolution levels such that the forward blur is consistent with
        applying the psf to the reconstructed pixel image. (default: True)
    normalize_psf: bool
        If True, the psf is normalized to sum to 1 (if ``restrict_psf`` is True, it is normalized
        at each resolution level. (default: True)
    resume: bool
        Resume iterations from a previous estimate? If True, the previous estimate must be provided
        via the ``output`` argument. (default: False)
    output: FloatParticles, optional
        Particle object to which the resulting values are written, or the initial estimate if
        ``resume`` is enabled. If not provided, a new object is generated (default: None)

    Returns
    -------
    output: FloatParticles
        The restored particle intensities.
    """

    _check_input(apr, parts, __allowed_input_types__)
    psf = __check_stencil(psf)
    if resume:
        _check_output(apr, output)
    output = output if isinstance(output, FloatParticles) else FloatParticles()
    _internals.richardson_lucy_tv(apr, parts, output, psf, num_iter, reg_param, restrict_psf, normalize_psf, resume)
    return output


def richardson_lucy_cuda(apr: APR,
                         parts: ParticleData,
                         psf: np.ndarray,
                         num_iter: int = 10,
                         restrict_psf: bool = True,
                         normalize_psf: bool = True,
                         resume: bool = False,
                         output: Optional[FloatParticles] = None) -> FloatParticles:
    """
    Richardson-Lucy deconvolution on the GPU. Requires pyapr to be built with CUDA support and ``psf`` to be of
    shape (3, 3, 3) or (5, 5, 5).

    Parameters
    ----------
    apr: APR
        Input APR object.
    parts: ByteParticles, ShortParticles or FloatParticles
        Input particle values.
    psf: np.ndarray
        Point spread function. Should be 3-dimensional and of type float32, otherwise
        it is expanded, e.g. shape (3, 3) -> (1, 3, 3), and cast.
    num_iter: int
        Number of iterations. (default: 10)
    restrict_psf: bool
        If True, the psf is adapted to coarser resolution levels such that the forward blur is consistent with
        applying the psf to the reconstructed pixel image. (default: True)
    normalize_psf: bool
        If True, the psf is normalized to sum to 1 (if ``restrict_psf`` is True, it is normalized
        at each resolution level. (default: True)
    resume: bool
        Resume iterations from a previous estimate? If True, the previous estimate must be provided
        via the ``output`` argument. (default: False)
    output: FloatParticles, optional
        Particle object to which the resulting values are written, or the initial estimate if
        ``resume`` is enabled. If not provided, a new object is generated (default: None)

    Returns
    -------
    output: FloatParticles
        The restored particle intensities.
    """

    _check_input(apr, parts, __allowed_input_types__)
    psf = __check_stencil(psf)
    if resume:
        _check_output(apr, output)
    output = output if isinstance(output, FloatParticles) else FloatParticles()

    if __cuda_build__:
        if psf.shape not in [(3, 3, 3), (5, 5, 5)]:
            raise ValueError(f'psf must be of shape (3, 3, 3) or (5, 5, 5), got {psf.shape}')
        _internals.richardson_lucy_cuda(apr, parts, output, psf, num_iter, restrict_psf, normalize_psf, resume)
    else:
        warn(f'richardson_lucy_cuda called but pyapr was not built with CUDA support. Using CPU version.')
        return richardson_lucy(apr, parts, psf, num_iter, restrict_psf, normalize_psf, resume, output)
    return output

