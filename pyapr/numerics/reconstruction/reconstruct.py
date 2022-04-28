from _pyaprwrapper.numerics.reconstruction import reconstruct_constant_inplace, \
                                                  reconstruct_level_inplace, \
                                                  reconstruct_smooth_inplace, \
                                                  reconstruct_constant_patch_inplace, \
                                                  reconstruct_level_patch_inplace, \
                                                  reconstruct_smooth_patch_inplace, \
                                                  reconstruct_constant_lazy_inplace, \
                                                  reconstruct_smooth_lazy_inplace, \
                                                  reconstruct_level_lazy_inplace
from _pyaprwrapper.data_containers import APR, ReconPatch, LazyIterator, \
                                          ShortParticles, LongParticles, FloatParticles, \
                                          LazyDataShort, LazyDataLong, LazyDataFloat
import numpy as np


def reconstruct_constant(apr: APR,
                         parts: (ShortParticles, LongParticles, FloatParticles),
                         tree_parts: (None, ShortParticles, LongParticles, FloatParticles) = None,
                         patch: (None, ReconPatch) = None,
                         out_arr: (None, np.ndarray) = None):
    """
    Reconstruct pixel values by piecewise constant interpolation

    Parameters
    ----------
    apr : APR
        input APR data structure
    parts : FloatParticles or ShortParticles
        input particle intensities
    tree_parts: None, FloatParticles or ShortParticles
        (optional) interior tree particle values used to construct at a lower resolution (if patch.level_delta < 0).
        If None, they are computed by average downsampling as necessary. (default: None)
    patch: ReconPatch
        (optional) specify the image region and resolution of the reconstruction. If None, reconstruct the full image volume
        at original pixel resolution. (default: None)
    out_arr: None, np.ndarray
        (optional) preallocated array for the result. If the size is not correct (according to APR dimensions or patch limits),
        memory for the output is reallocated. (default: None)
    Returns
    -------
    out_arr : numpy.ndarray
        the reconstructed pixel values
    """

    if isinstance(parts, FloatParticles):
        _dtype = np.float32
    elif isinstance(parts, LongParticles):
        _dtype = np.uint64
    elif isinstance(parts, ShortParticles):
        _dtype = np.uint16
    else:
        raise ValueError('parts type not recognized')

    if patch is not None:
        if not patch.check_limits(apr):
            return None

        if out_arr is None or out_arr.size != patch.size():
            out_arr = np.zeros(shape=(patch.z_end-patch.z_begin, patch.x_end-patch.x_begin, patch.y_end-patch.y_begin),
                               dtype=_dtype)

        if tree_parts is None:
            tree_parts = FloatParticles()

        reconstruct_constant_patch_inplace(apr, parts, tree_parts, patch, out_arr)
    else:
        _shape = [apr.org_dims(2), apr.org_dims(1), apr.org_dims(0)]
        if out_arr is None or out_arr.size != np.prod(_shape):
            out_arr = np.zeros(shape=_shape, dtype=_dtype)

        reconstruct_constant_inplace(apr, parts, out_arr)
    return out_arr


def reconstruct_smooth(apr: APR,
                       parts: (ShortParticles, LongParticles, FloatParticles),
                       tree_parts: (None, ShortParticles, LongParticles, FloatParticles) = None,
                       patch: (None, ReconPatch) = None,
                       out_arr: (None, np.ndarray) = None):
    """
    Reconstruct pixel values by smooth interpolation

    Parameters
    ----------
    apr : APR
        input APR data structure
    parts : FloatParticles or ShortParticles
        input particle intensities
    tree_parts: None, FloatParticles or ShortParticles
        (optional) interior tree particle values used to construct at a lower resolution (if patch.level_delta < 0).
        If None, they are computed by average downsampling as necessary. (default: None)
    patch: ReconPatch
        (optional) specify the image region and resolution of the reconstruction. If None, reconstruct the full image volume
        at original pixel resolution. (default: None)
    out_arr: None, np.ndarray
        (optional) preallocated array for the result. If the size is not correct (according to APR dimensions or patch limits),
        memory for the output is reallocated. (default: None)
    Returns
    -------
    out_arr : numpy.ndarray
        the reconstructed pixel values
    """

    if isinstance(parts, FloatParticles):
        _dtype = np.float32
    elif isinstance(parts, LongParticles):
        _dtype = np.uint64
    elif isinstance(parts, ShortParticles):
        _dtype = np.uint16
    else:
        raise ValueError('parts type not recognized')

    if patch is not None:
        if not patch.check_limits(apr):
            return None

        if out_arr is None or out_arr.size != patch.size():
            out_arr = np.zeros(shape=(patch.z_end-patch.z_begin, patch.x_end-patch.x_begin, patch.y_end-patch.y_begin),
                               dtype=_dtype)

        if tree_parts is None:
            tree_parts = FloatParticles()

        reconstruct_smooth_patch_inplace(apr, parts, tree_parts, patch, out_arr)
    else:
        _shape = [apr.org_dims(2), apr.org_dims(1), apr.org_dims(0)]
        if out_arr is None or out_arr.size != np.prod(_shape):
            out_arr = np.zeros(shape=_shape, dtype=_dtype)

        reconstruct_smooth_inplace(apr, parts, out_arr)
    return out_arr


def reconstruct_level(apr: APR,
                      patch: (None, ReconPatch) = None,
                      out_arr: (None, np.ndarray) = None):
    """
    Construct pixel values containing the level of the particle at the corresponding location.

    Parameters
    ----------
    apr : APR
        input APR data structure
    patch: ReconPatch
        (optional) specify the image region and resolution of the reconstruction. If None, reconstruct the full image volume
        at original pixel resolution. (default: None)
    out_arr: None, np.ndarray
        (optional) preallocated array for the result. If the size is not correct (according to APR dimensions or patch limits),
        memory for the output is reallocated. (default: None)
    Returns
    -------
    out_arr : numpy.ndarray
        the reconstructed pixel values
    """

    if patch is not None:
        if not patch.check_limits(apr):
            return None

        if out_arr is None or out_arr.size != patch.size():
            out_arr = np.zeros(shape=(patch.z_end-patch.z_begin, patch.x_end-patch.x_begin, patch.y_end-patch.y_begin),
                               dtype=np.uint8)

        reconstruct_level_patch_inplace(apr, patch, out_arr)
    else:
        _shape = [apr.org_dims(2), apr.org_dims(1), apr.org_dims(0)]
        if out_arr is None or out_arr.size == np.prod(_shape):
            out_arr = np.zeros(shape=_shape, dtype=np.uint8)

        reconstruct_level_inplace(apr, out_arr)
    return out_arr


def reconstruct_constant_lazy(apr_it: LazyIterator,
                              tree_it: LazyIterator,
                              parts: (LazyDataShort, LazyDataLong, LazyDataFloat),
                              tree_parts: (LazyDataShort, LazyDataLong, LazyDataFloat),
                              patch: ReconPatch,
                              out_arr: (None, np.ndarray) = None):

    if isinstance(parts, LazyDataFloat):
        _dtype = np.float32
    elif isinstance(parts, LazyDataLong):
        _dtype = np.uint64
    elif isinstance(parts, LazyDataShort):
        _dtype = np.uint16
    else:
        raise ValueError('parts type not recognized')

    if out_arr is None or out_arr.size != patch.size():
        out_arr = np.zeros(shape=(patch.z_end-patch.z_begin, patch.x_end-patch.x_begin, patch.y_end-patch.y_begin),
                           dtype=_dtype)

    reconstruct_constant_lazy_inplace(apr_it, tree_it, out_arr, parts, tree_parts, patch)
    return out_arr


def reconstruct_level_lazy(apr_it: LazyIterator,
                           tree_it: LazyIterator,
                           patch: ReconPatch,
                           out_arr: (None, np.ndarray) = None):

    _dtype = np.uint8

    if out_arr is None or out_arr.size != patch.size():
        out_arr = np.zeros(shape=(patch.z_end-patch.z_begin, patch.x_end-patch.x_begin, patch.y_end-patch.y_begin),
                           dtype=_dtype)

    reconstruct_level_lazy_inplace(apr_it, tree_it, out_arr, patch)
    return out_arr


def reconstruct_smooth_lazy(apr_it: LazyIterator,
                            tree_it: LazyIterator,
                            parts: (LazyDataShort, LazyDataLong, LazyDataFloat),
                            tree_parts: (LazyDataShort, LazyDataLong, LazyDataFloat),
                            patch: ReconPatch,
                            out_arr: (None, np.ndarray) = None):

    if isinstance(parts, LazyDataFloat):
        _dtype = np.float32
    elif isinstance(parts, LazyDataLong):
        _dtype = np.uint64
    elif isinstance(parts, LazyDataShort):
        _dtype = np.uint16
    else:
        raise ValueError('parts type not recognized')

    if out_arr is None or out_arr.size != patch.size():
        out_arr = np.zeros(shape=(patch.z_end-patch.z_begin, patch.x_end-patch.x_begin, patch.y_end-patch.y_begin),
                           dtype=_dtype)

    reconstruct_smooth_lazy_inplace(apr_it, tree_it, out_arr, parts, tree_parts, patch)
    return out_arr
