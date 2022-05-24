from _pyaprwrapper.reconstruction import reconstruct_constant_inplace, \
                                         reconstruct_level_inplace, \
                                         reconstruct_smooth_inplace, \
                                         reconstruct_constant_patch_inplace, \
                                         reconstruct_level_patch_inplace, \
                                         reconstruct_smooth_patch_inplace, \
                                         reconstruct_constant_lazy_inplace, \
                                         reconstruct_smooth_lazy_inplace, \
                                         reconstruct_level_lazy_inplace
from _pyaprwrapper.data_containers import APR, ReconPatch, LazyIterator, LazyAccess, \
                                          ByteParticles, ShortParticles, LongParticles, FloatParticles, \
                                          LazyDataByte, LazyDataShort, LazyDataLong, LazyDataFloat
from ..io import APRFile, get_particle_type
from ..utils import type_to_lazy_particles, particles_to_type
from .._common import _check_input
import numpy as np
from typing import Optional, Union


ParticleData = Union[ByteParticles, ShortParticles, FloatParticles, LongParticles]
LazyData = Union[LazyDataByte, LazyDataShort, LazyDataFloat, LazyDataLong]


__all__ = ['reconstruct_constant',
           'reconstruct_smooth',
           'reconstruct_level',
           'reconstruct_constant_lazy',
           'reconstruct_smooth_lazy',
           'reconstruct_level_lazy',
           'reconstruct_lazy']


def reconstruct_constant(apr: APR,
                         parts: ParticleData,
                         tree_parts: Optional[ParticleData] = None,
                         patch: Optional[ReconPatch] = None,
                         out_arr: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Reconstruct pixel values by piecewise constant interpolation

    Parameters
    ----------
    apr : APR
        input APR data structure
    parts : ParticleData
        input particle intensities
    tree_parts: ParticleData, optional
        (optional) interior tree particle values used to construct at a lower resolution (if patch.level_delta < 0).
        If None, they are computed by average downsampling as necessary. (default: None)
    patch: ReconPatch, optional
        (optional) specify the image region and resolution of the reconstruction. If None, reconstruct the full image volume
        at original pixel resolution. (default: None)
    out_arr: numpy.ndarray, optional
        (optional) preallocated array for the result. If the size is not correct (according to APR dimensions or patch limits),
        memory for the output is reallocated. (default: None)
    Returns
    -------
    out_arr : numpy.ndarray
        The reconstructed pixel values.
    """
    _check_input(apr, parts)
    _dtype = particles_to_type(parts)

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
                       parts: ParticleData,
                       tree_parts: Optional[ParticleData] = None,
                       patch: Optional[ReconPatch] = None,
                       out_arr: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Reconstruct pixel values by smooth interpolation

    Parameters
    ----------
    apr : APR
        input APR data structure
    parts : ParticleData
        input particle intensities
    tree_parts: ParticleData, optional
        (optional) interior tree particle values used to construct at a lower resolution (if patch.level_delta < 0).
        If None, they are computed by average downsampling as necessary. (default: None)
    patch: ReconPatch, optional
        (optional) specify the image region and resolution of the reconstruction. If None, reconstruct the full image volume
        at original pixel resolution. (default: None)
    out_arr: numpy.ndarray, optional
        (optional) preallocated array for the result. If the size is not correct (according to APR dimensions or patch limits),
        memory for the output is reallocated. (default: None)
    Returns
    -------
    out_arr : numpy.ndarray
        The reconstructed pixel values.
    """
    _check_input(apr, parts)
    _dtype = particles_to_type(parts)

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
                      patch: Optional[ReconPatch] = None,
                      out_arr: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Construct pixel values containing the level of the particle at the corresponding location.

    Parameters
    ----------
    apr : APR
        input APR data structure
    patch: ReconPatch, optional
        (optional) specify the image region and resolution of the reconstruction. If None, reconstruct the full image volume
        at original pixel resolution. (default: None)
    out_arr: numpy.ndarray, optional
        (optional) preallocated array for the result. If the size is not correct (according to APR dimensions or patch limits),
        memory for the output is reallocated. (default: None)
    Returns
    -------
    out_arr : numpy.ndarray
        The reconstructed pixel values.
    """
    assert apr.total_number_particles() > 0, ValueError(f'APR not initialized!')
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
                              parts: LazyData,
                              tree_parts: LazyData,
                              patch: ReconPatch,
                              out_arr: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Lazy constant reconstruction of an image region.

    Parameters
    ----------
    apr_it: LazyIterator
        Lazy iterator for APR structure, must be initialized and have the file open.
    tree_it: LazyIterator
        Lazy iterator for tree structure, must be initialized and have the file open.
    parts : LazyData
        LazyData object for APR particle values, must be initialized and have the file open.
    tree_parts : LazyData
        LazyData object for tree particle values, must be initialized and have the file open.
    patch: ReconPatch
        Specify the image region and resolution of the reconstruction.
    out_arr: numpy.ndarray, optional
        (optional) preallocated array for the result. If the size is not correct (according to APR dimensions or
        patch limits), memory for the output is reallocated. (default: None)

    Returns
    -------
    out_arr : numpy.ndarray
        The reconstructed pixel values.
    """

    _dtype = particles_to_type(parts)

    if out_arr is None or out_arr.size != patch.size():
        out_arr = np.zeros(shape=(patch.z_end-patch.z_begin, patch.x_end-patch.x_begin, patch.y_end-patch.y_begin),
                           dtype=_dtype)

    reconstruct_constant_lazy_inplace(apr_it, tree_it, parts, tree_parts, patch, out_arr)
    return out_arr


def reconstruct_level_lazy(apr_it: LazyIterator,
                           tree_it: LazyIterator,
                           patch: ReconPatch,
                           out_arr: (None, np.ndarray) = None) -> np.ndarray:
    """
    Lazy level reconstruction of an image region. Each pixel in the output takes the value of the
    resolution level of the particle at the corresponding location.

    Parameters
    ----------
    apr_it: LazyIterator
        Lazy iterator for APR structure, must be initialized and have the file open.
    tree_it: LazyIterator
        Lazy iterator for tree structure, must be initialized and have the file open.
    patch: ReconPatch
        Specify the image region and resolution of the reconstruction.
    out_arr: numpy.ndarray, optional
        (optional) preallocated array for the result. If the size is not correct (according to APR dimensions or
        patch limits), memory for the output is reallocated. (default: None)

    Returns
    -------
    out_arr : numpy.ndarray
        The reconstructed pixel values.
    """

    if out_arr is None or out_arr.size != patch.size():
        out_arr = np.zeros(shape=(patch.z_end-patch.z_begin, patch.x_end-patch.x_begin, patch.y_end-patch.y_begin),
                           dtype=np.uint8)

    reconstruct_level_lazy_inplace(apr_it, tree_it, patch, out_arr)
    return out_arr


def reconstruct_smooth_lazy(apr_it: LazyIterator,
                            tree_it: LazyIterator,
                            parts: LazyData,
                            tree_parts: LazyData,
                            patch: ReconPatch,
                            out_arr: (None, np.ndarray) = None) -> np.ndarray:
    """
    Lazy smooth reconstruction of an image region.

    Parameters
    ----------
    apr_it: LazyIterator
        Lazy iterator for APR structure, must be initialized and have the file open.
    tree_it: LazyIterator
        Lazy iterator for tree structure, must be initialized and have the file open.
    parts : LazyData
        LazyData object for APR particle values, must be initialized and have the file open.
    tree_parts : LazyData
        LazyData object for tree particle values, must be initialized and have the file open.
    patch: ReconPatch
        Specify the image region and resolution of the reconstruction.
    out_arr: numpy.ndarray, optional
        (optional) preallocated array for the result. If the size is not correct (according to APR dimensions or
        patch limits), memory for the output is reallocated. (default: None)

    Returns
    -------
    out_arr : numpy.ndarray
        The reconstructed pixel values.
    """

    _dtype = particles_to_type(parts)

    if out_arr is None or out_arr.size != patch.size():
        out_arr = np.zeros(shape=(patch.z_end-patch.z_begin, patch.x_end-patch.x_begin, patch.y_end-patch.y_begin),
                           dtype=_dtype)

    reconstruct_smooth_lazy_inplace(apr_it, tree_it, parts, tree_parts, patch, out_arr)
    return out_arr


def reconstruct_lazy(file_path: str,
                     patch: Optional[ReconPatch] = None,
                     mode: str = 'constant',
                     t: int = 0,
                     channel_name: str = 't',
                     parts_name: str = 'particles',
                     tree_parts_name: str = 'particles',
                     out_arr: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Lazy reconstruction of an image (region) directly from a given APR file.

    Parameters
    ----------
    file_path: str
        APR file path, e.g. '/home/data/test.apr'.
    patch: ReconPatch, optional
        (optional) Specify the image region and resolution of the reconstruction. If `None`, the entire volume is
        reconstructed at the original pixel resolution. (default: None)
    mode: str
        Reconstruction mode to use. Allowed values are 'constant', 'smooth', or 'level'. (default: 'constant')
    t: int
        Time point to open in the file. (default: 0)
    channel_name: str
        Channel to open in the file. (default: 't')
    parts_name: str
        Name of the particle value field to read. Only used for modes 'constant' and 'smooth'. (default: 'particles')
    tree_parts_name: str
        Name of the tree particle value field to read. Only used if `patch.level_delta < 0` for modes 'constant'
        and 'smooth'. (default: 'particles')
    out_arr: numpy.ndarray, optional
        (optional) Pre-allocated array for the result. If the size is not correct (according to APR dimensions or
        patch limits), memory for the output is reallocated. (default: None)

    Returns
    -------
    out_arr: numpy.ndarray
        The reconstructed pixel values.
    """

    apr_file = APRFile()
    apr_file.open(file_path, 'READ')

    # initialize lazy access
    access = LazyAccess()
    access.init(apr_file)
    access.open()
    apr_it = LazyIterator(access)

    # initialize lazy data
    if mode != 'level':
        parts_type = get_particle_type(file_path, t=t, channel_name=channel_name, parts_name=parts_name, tree=False)
        parts = type_to_lazy_particles(parts_type)
        parts.init(apr_file, parts_name, t, channel_name)
        parts.open()
        _dtype = np.float32 if parts_type == 'float' else np.dtype(parts_type)
        tree_parts = type(parts)()
    else:
        _dtype = np.uint8

    # instantiate tree iterator and data (only used if patch.level_delta < 0)
    tree_access = LazyAccess()
    tree_it = LazyIterator()

    patch = patch or ReconPatch()

    if patch.level_delta < 0:
        tree_access.init_tree(apr_file)
        tree_access.open()
        tree_it = LazyIterator(tree_access)

        if mode != 'level':
            tree_parts.init_tree(apr_file, tree_parts_name, t, channel_name)
            tree_parts.open()

    if not patch.check_limits(access):
        return None

    if out_arr is None or out_arr.size != patch.size():
        out_arr = np.zeros(shape=(patch.z_end-patch.z_begin, patch.x_end-patch.x_begin, patch.y_end-patch.y_begin),
                           dtype=_dtype)

    if mode == 'constant':
        reconstruct_constant_lazy_inplace(apr_it, tree_it, parts, tree_parts, patch, out_arr)
    elif mode == 'smooth':
        reconstruct_smooth_lazy_inplace(apr_it, tree_it, parts, tree_parts, patch, out_arr)
    elif mode == 'level':
        reconstruct_level_lazy_inplace(apr_it, tree_it, patch, out_arr)
    else:
        raise ValueError(f'mode \'{mode}\' not recognized - allowed values are \'constant\', \'smooth\' and \'level\'')

    # close open file(s) and return
    if patch.level_delta < 0:
        tree_access.close()
        if mode != 'level':
            tree_parts.close()
    if mode != 'level':
        parts.close()
    access.close()
    apr_file.close()

    return out_arr
