from _pyaprwrapper.numerics.transform import erosion, dilation, remove_small_objects, remove_large_objects, \
                                             find_objects_cpp, find_label_centers_cpp, find_label_centers_weighted_cpp, \
                                             find_label_volume_cpp
from _pyaprwrapper.numerics.segmentation import connected_component
from _pyaprwrapper.data_containers import APR, ShortParticles, LongParticles, FloatParticles
import numpy as np
from typing import Optional, Union, Tuple


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
    erosion(apr, parts_copy, binary=binary, radius=radius)
    dilation(apr, parts_copy, binary=binary, radius=radius)
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
    dilation(apr, parts_copy, binary=binary, radius=radius)
    erosion(apr, parts_copy, binary=binary, radius=radius)
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
    remove_small_objects(apr, cc_inverted, min_volume=min_volume)
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


def find_objects(apr: APR,
                 labels: Union[ShortParticles, LongParticles]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find and return tight bounding boxes for each unique input label. Assumes that the labels are
    ordered from 0, such that 0 is background and each value > 0 corresponds to a connected component.

    Parameters
    ----------
    apr: APR
        Input APR data structure.
    labels: ShortParticles or LongParticles
        Input (object) labels.

    Returns
    -------
    min_coords: numpy.ndarray
        Array of shape `(labels.max() + 1, 3)` containing the "lower" corner of each bounding box in z, x and y.
    max_coords: numpy.ndarray
        Array of shape `(labels.max() + 1, 3)` containing the "upper" corner of each bounding box in z, x and y.
    """

    max_label = labels.max()
    max_dim = max([apr.org_dims(x) for x in range(3)])
    min_coords = np.full((max_label+1, 3), max_dim+1, dtype=np.int32)
    max_coords = np.zeros((max_label+1, 3), dtype=np.int32)
    find_objects_cpp(apr, labels, min_coords, max_coords)

    max_coords[0, :] = [apr.org_dims(x) for x in [2, 0, 1]]
    min_coords[0, :] = 0

    return min_coords, max_coords


def find_label_centers(apr: APR,
                       labels: Union[ShortParticles, LongParticles],
                       weights: Optional[Union[ShortParticles, FloatParticles]] = None) -> np.ndarray:
    """
    Compute the volumetric center of each unique input label, optionally weighted by, e.g., image intensity.

    Parameters
    ----------
    apr: APR
        Input APR data structure.
    labels: ShortParticles or LongParticles
        Input (object) labels.
    weights: ShortParticles or FloatParticles, optional
        (optional) Weight for each particle. Normalization is applied internally. (default: None)

    Returns
    -------
    coords: numpy.ndarray
        Array containing the center coordinates.
    """

    max_label = labels.max()
    coords = np.zeros((max_label+1, 3), dtype=np.float64)
    if weights is not None:
        find_label_centers_weighted_cpp(apr, labels, coords, weights)
    else:
        find_label_centers_cpp(apr, labels, coords)
    return coords[np.any(coords > 0, axis=1), :]


def find_label_volume(apr: APR,
                      labels: Union[ShortParticles, LongParticles]) -> np.ndarray:
    """
    Return the volume (in voxels) of each unique input label.

    Parameters
    ----------
    apr: APR
        Input APR data structure.
    labels: ShortParticles or LongParticles
        Input (object) labels.

    Returns
    -------
    volume: numpy.ndarray
        Array of shape `(labels.max() + 1,)` containing the label volumes.
    """

    max_label = labels.max()
    volume = np.zeros((max_label+1), dtype=np.uint64)
    find_label_volume_cpp(apr, labels, volume)
    return volume
