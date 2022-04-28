from _pyaprwrapper.numerics.transform import *
from _pyaprwrapper.data_containers import APR, ReconPatch, ShortParticles, FloatParticles
import numpy as np
from warnings import warn
from typing import Optional, Union


def maximum_projection(apr: APR,
                       parts: Union[ShortParticles, FloatParticles],
                       dim: int,
                       patch: Optional[ReconPatch] = None,
                       method: str = 'auto'):
    """
    Compute the maximum intensity projection along an axis.

    Note: assumes that all particle values are non-negative

    Parameters
    ----------
    apr: APR
        Input APR data structure
    parts: FloatParticles or ShortParticles
        Input particle intensities
    dim: int
        Dimension along which to compute the projection:
        `dim=0`: project along Y to produce a ZX plane
        `dim=1`: project along X to produce a ZY plane
        `dim=2`: project along Z to produce an XY plane
    patch: ReconPatch, optional
        (optional) If provided, projects only within the image region specified by `patch`. Otherwise projects
        through the entire volume. (default: None)
    method: str
        Specify the projection algorithm (results are identical, but performance may differ). Supported arguments
        are `auto`, `direct` and `pyramid`.

    Returns
    -------
    out : numpy.ndarray
        The computed maximum intensity projection
    """

    if dim not in (0, 1, 2):
        raise ValueError("dim must be 0, 1 or 2 corresponding to projection along y, x or z")

    if not isinstance(parts, (ShortParticles, FloatParticles)):
        raise TypeError("input particles must be ShortParticles or FloatParticles")

    args = (apr, parts)

    if patch is not None:
        if patch.level_delta != 0:
            warn('max projection is not yet implemented for patch.level_delta != 0. '
                 'Proceeding with level_delta = 0.', RuntimeWarning)

        # temporarily set level_delta to 0 TODO: make it allow non-zero level delta
        tmp = patch.level_delta
        patch.level_delta = 0
        if not patch.check_limits(apr):
            return None
        patch.level_delta = tmp
        args += (patch,)

    if dim == 0:
        return np.array(max_projection_y(*args), copy=False).squeeze() if method in ('direct', 'auto') \
            else np.array(max_projection_y_alt(*args), copy=False).squeeze()
    elif dim == 1:
        return np.array(max_projection_x(*args), copy=False).squeeze() if method == 'direct' \
            else np.array(max_projection_x_alt(*args), copy=False).squeeze()
    else:
        return np.array(max_projection_z(*args), copy=False).squeeze() if method == 'direct' \
            else np.array(max_projection_z_alt(*args), copy=False).squeeze()
