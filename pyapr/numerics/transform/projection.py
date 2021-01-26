import pyapr
import numpy as np


def maximum_projection(apr: pyapr.APR,
                       parts: (pyapr.ShortParticles, pyapr.FloatParticles),
                       dim: int):
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

    Returns
    -------
    out : numpy.ndarray
        the computed maximum intensity projection
    """

    if dim not in (0, 1, 2):
        raise ValueError("dim must be 0, 1 or 2 corresponding to projection along y, x or z")

    if not isinstance(parts, (pyapr.ShortParticles, pyapr.FloatParticles)):
        raise TypeError("input particles must be ShortParticles or FloatParticles")

    if dim == 0:
        out = np.zeros((apr.z_num(apr.level_max()), apr.x_num(apr.level_max())), dtype=np.float32)
        pyapr.numerics.transform.max_projection_y(apr, parts, out)
    elif dim == 1:
        out = np.zeros((apr.z_num(apr.level_max()), apr.y_num(apr.level_max())), dtype=np.float32)
        pyapr.numerics.transform.max_projection_x(apr, parts, out)
    else:
        out = np.zeros((apr.x_num(apr.level_max()), apr.y_num(apr.level_max())), dtype=np.float32)
        pyapr.numerics.transform.max_projection_z(apr, parts, out)

    return out
