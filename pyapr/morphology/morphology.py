import _pyaprwrapper.morphology as _internals # import erosion, binary_erosion, dilation, binary_ remove_small_objects, remove_large_objects
from _pyaprwrapper.data_containers import APR, ByteParticles, ShortParticles, LongParticles, FloatParticles
import numpy as np
from typing import Union


def opening(apr: APR,
            parts: Union[ShortParticles, LongParticles, FloatParticles],
            binary: bool = False,
            radius: int = 1,
            inplace: bool = False) -> Union[ShortParticles, LongParticles, FloatParticles]:
    """
    Apply morphological opening (erosion followed by dilation) to an APR image. Only considers face-side neighbors
    (6-connectivity in 3D). If `radius > 1` the erosion and dilation are repeated `radius` times.

    Parameters
    ----------
    apr: APR
        Input APR data structure.
    parts: ShortParticles, LongParticles, FloatParticles
        Input particle values.
    binary: bool
        If `True`, binary erosion and dilation is used. Otherwise `min` and `max` operators are used. (default: False)
    radius: int
        Radius of the erosion and dilation operations. The 6-connective operations are iterated `radius` times. (default: 1)
    inplace: bool
        If `True`, the operation modifies the input `parts`. Otherwise, a copy of the input is used. (default: False)

    Returns
    -------
    parts_copy: ShortParticles, LongParticles, FloatParticles
        Output particle values of the same type as the input `parts`.
    """

    parts_copy = parts if inplace else parts.copy()
    if binary:
        _internals.binary_erosion(apr, parts_copy, radius)
        _internals.binary_dilation(apr, parts_copy, radius)
    else:
        _internals.erosion(apr, parts_copy, radius)
        _internals.dilation(apr, parts_copy, radius)
    return parts_copy


def closing(apr: APR,
            parts: Union[ShortParticles, LongParticles, FloatParticles],
            binary: bool = False,
            radius: int = 1,
            inplace: bool = False) -> Union[ShortParticles, LongParticles, FloatParticles]:
    """
    Apply morphological closing (dilation followed by erosion) to an APR image. Only considers face-side neighbors
    (6-connectivity in 3D). If `radius > 1` the erosion and dilation are repeated `radius` times.

    Parameters
    ----------
    apr: APR
        Input APR data structure.
    parts: ShortParticles, LongParticles, FloatParticles
        Input particle values.
    binary: bool
        If `True`, binary erosion and dilation is used. Otherwise `min` and `max` operators are used. (default: False)
    radius: int
        Radius of the erosion and dilation operations. The 6-connective operations are iterated `radius` times. (default: 1)
    inplace: bool
        If `True`, the operation modifies the input `parts`. Otherwise, a copy of the input is used. (default: False)

    Returns
    -------
    parts_copy: ShortParticles, LongParticles, FloatParticles
        Output particle values of the same type as the input `parts`.
    """

    parts_copy = parts if inplace else parts.copy()
    _internals.dilation(apr, parts_copy, binary=binary, radius=radius)
    _internals.erosion(apr, parts_copy, binary=binary, radius=radius)
    return parts_copy


def tophat(apr: APR,
           parts: Union[ShortParticles, LongParticles, FloatParticles],
           binary: bool = False,
           radius: int = 1) -> Union[ShortParticles, LongParticles, FloatParticles]:
    """
    Apply top-hat transform to an APR image by subtracting an opening of the image.

    Parameters
    ----------
    apr: APR
        Input APR data structure.
    parts: ShortParticles, LongParticles, FloatParticles
        Input particle values.
    binary: bool
        Use binary opening? (default: False)
    radius: int
        Radius of the opening operation. (default: 1)

    Returns
    -------
    out: ShortParticles, LongParticles, FloatParticles
        Output particle values of the same type as the input `parts`.

    See also
    --------
    pyapr.numerics.transform.opening
    """

    tmp = opening(apr, parts, binary=binary, radius=radius, inplace=False)
    return parts - tmp


def bottomhat(apr: APR,
              parts: Union[ShortParticles, LongParticles, FloatParticles],
              binary: bool = False,
              radius: int = 1) -> Union[ShortParticles, LongParticles, FloatParticles]:
    """
    Apply bottom-hat transform to an APR image by subtracting it from a closing of itself.

    Parameters
    ----------
    apr: APR
        Input APR data structure.
    parts: ShortParticles, LongParticles, FloatParticles
        Input particle values.
    binary: bool
        Use binary closing? (default: False)
    radius: int
        Radius of the closing operation. (default: 1)

    Returns
    -------
    out: ShortParticles, LongParticles, FloatParticles
        Output particle values of the same type as the input `parts`.

    See also
    --------
    pyapr.numerics.transform.closing
    """

    tmp = closing(apr, parts, binary=binary, radius=radius, inplace=False)
    return tmp - parts


def remove_small_holes(apr: APR,
                       labels: Union[ShortParticles, LongParticles],
                       min_volume: int = 200) -> Union[ShortParticles, LongParticles]:
    """
    Remove holes smaller than a threshold from an input particle mask. Assumes that the background value is 0.

    Parameters
    ----------
    apr: APR
        Input APR data structure.
    labels: ShortParticles, LongParticles
        Input particle mask.
    min_volume: int
        Remove holes smaller in volume than `min_volume` voxels. (default: 200)

    Returns
    -------
    out: ShortParticles, LongParticles
        The mask with holes removed. If the input mask was binary, a binary mask is returned. Otherwise, connected
        components are recomputed.
    """

    mask = labels < 1
    cc_inverted = ShortParticles()
    connected_component(apr, mask, cc_inverted)
    _internals.remove_small_objects(apr, cc_inverted, min_volume=min_volume)
    mask = cc_inverted < 1

    if labels.max() > 1:
        # Case where input is a label map
        if isinstance(labels, ShortParticles):
            cc = ShortParticles()
        elif isinstance(labels, LongParticles):
            mask = np.array(mask).astype('uint16')
            mask = ShortParticles(mask)
            cc = LongParticles()
        connected_component(apr, mask, cc)
        return cc
    return mask
