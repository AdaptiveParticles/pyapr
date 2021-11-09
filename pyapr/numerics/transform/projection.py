import pyapr
import numpy as np
from warnings import warn


def maximum_projection(apr: pyapr.APR,
                       parts: (pyapr.ShortParticles, pyapr.FloatParticles),
                       dim: int,
                       patch: (pyapr.ReconPatch, None) = None,
                       method: str = 'auto'):
    """
    Compute the maximum intensity projection along an axis.

    Note: assumes that all particle values are non-negative

    Parameters
    ----------
    apr : pyapr.APR
        input APR data structure
    parts : pyapr.FloatParticles or pyapr.ShortParticles
        input particle intensities
    dim : int
        dimension along which to compute the projection.
        dim=0: project along Y to produce a ZX plane
        dim=1: project along X to produce a ZY plane
        dim=2: project along Z to produce an XY plane
    patch : pyapr.ReconPatch
        if provided, projects only within the image region specified by `patch`.
    method:
        specify the projection algorithm (results are identical, but performance may differ). Supported arguments
        are `auto`, `direct` and `pyramid`.

    Returns
    -------
    out : numpy.ndarray
        the computed maximum intensity projection
    """

    if dim not in (0, 1, 2):
        raise ValueError("dim must be 0, 1 or 2 corresponding to projection along y, x or z")

    if not isinstance(parts, (pyapr.ShortParticles, pyapr.FloatParticles)):
        raise TypeError("input particles must be ShortParticles or FloatParticles")

    args = (apr, parts)

    if patch is not None:
        if patch.level_delta != 0:
            print('Warning: patched max projection is not yet implemented for level_delta != 0. Proceeding with level_delta = 0.')

        # temporarily set level_delta to 0 TODO: make it allow non-zero level delta
        tmp = patch.level_delta
        patch.level_delta = 0
        if not patch.check_limits(apr):
            return None
        patch.level_delta = tmp
        args += (patch,)

    if dim == 0:
        return np.array(pyapr.numerics.transform.max_projection_y(*args), copy=False).squeeze() if method in ('direct', 'auto') \
            else np.array(pyapr.numerics.transform.max_projection_y_alt(*args), copy=False).squeeze()
    elif dim == 1:
        return np.array(pyapr.numerics.transform.max_projection_x(*args), copy=False).squeeze() if method == 'direct' \
            else np.array(pyapr.numerics.transform.max_projection_x_alt(*args), copy=False).squeeze()
    else:
        return np.array(pyapr.numerics.transform.max_projection_z(*args), copy=False).squeeze() if method == 'direct' \
            else np.array(pyapr.numerics.transform.max_projection_z_alt(*args), copy=False).squeeze()


def maximum_projection_patch(apr: pyapr.APR,
                             parts: (pyapr.ShortParticles, pyapr.FloatParticles),
                             dim: int,
                             patch: pyapr.ReconPatch,
                             method: str = 'auto'):
    warn('\'maximum_projection_patch\' is deprecated and will be removed in a future release. '
         'Returning \'maximum_projection\' with the same arguments.', DeprecationWarning)
    return maximum_projection(apr, parts, dim, patch, method)
