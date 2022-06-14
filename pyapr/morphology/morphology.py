import _pyaprwrapper.morphology as _internals
from _pyaprwrapper.data_containers import APR, ByteParticles, ShortParticles, LongParticles, FloatParticles
from _pyaprwrapper.measure import connected_component
from .._common import _check_input
from typing import Union, Optional

__allowed_label_types__ = (ByteParticles, ShortParticles, LongParticles)
ParticleData = Union[ByteParticles, ShortParticles, LongParticles, FloatParticles]


def dilation(apr: APR,
             parts: ParticleData,
             radius: int = 1,
             binary: bool = False,
             inplace: bool = False) -> ParticleData:
    """
    Apply morphological dilation (binary or grayscale) to an APR image. Only considers face-side neighbors
    (6-connectivity in 3D). If ``radius > 1`` the operation is repeated ``radius`` times.

    Parameters
    ----------
    apr: APR
        Input APR data structure.
    parts: ByteParticles, ShortParticles, LongParticles or FloatParticles
        Input particle values.
    binary: bool
        If `True`, apply binary dilation. Otherwise, grayscale dilation is used. (default: False)
    radius: int
        Radius of the operation (the 6-connective operation is iterated ``radius`` times). (default: 1)
    inplace: bool
        If `True`, the operation modifies the input ``parts``. Otherwise, a copy of the input is used. (default: False)

    Returns
    -------
    parts_copy: ByteParticles, ShortParticles, LongParticles or FloatParticles
        Output particle values of the same type as the input ``parts``.
    """
    _check_input(apr, parts)
    parts_copy = parts if inplace else parts.copy()
    if binary:
        _internals.binary_dilation(apr, parts_copy, radius)
    else:
        _internals.dilation(apr, parts_copy, radius)
    return parts_copy


def erosion(apr: APR,
            parts: ParticleData,
            radius: int = 1,
            binary: bool = False,
            inplace: bool = False) -> ParticleData:
    """
    Apply morphological erosion (binary or grayscale) to an APR image. Only considers face-side neighbors
    (6-connectivity in 3D). If ``radius > 1`` the operation is repeated ``radius`` times.

    Parameters
    ----------
    apr: APR
        Input APR data structure.
    parts: ByteParticles, ShortParticles, LongParticles or FloatParticles
        Input particle values.
    binary: bool
        If `True`, apply binary erosion. Otherwise, grayscale erosion is used. (default: False)
    radius: int
        Radius of the operation (the 6-connective operation is iterated ``radius`` times). (default: 1)
    inplace: bool
        If `True`, the operation modifies the input ``parts``. Otherwise, a copy of the input is used. (default: False)

    Returns
    -------
    parts_copy: ByteParticles, ShortParticles, LongParticles or FloatParticles
        Output particle values of the same type as the input ``parts``.
    """
    _check_input(apr, parts)
    parts_copy = parts if inplace else parts.copy()
    if binary:
        _internals.binary_erosion(apr, parts_copy, radius)
    else:
        _internals.erosion(apr, parts_copy, radius)
    return parts_copy


def opening(apr: APR,
            parts: ParticleData,
            binary: bool = False,
            radius: int = 1,
            inplace: bool = False) -> ParticleData:
    """
    Apply morphological opening (erosion followed by dilation) to an APR image. Only considers face-side neighbors
    (6-connectivity in 3D). If ``radius > 1`` the erosion and dilation are repeated ``radius`` times.

    Parameters
    ----------
    apr: APR
        Input APR data structure.
    parts: ByteParticles, ShortParticles, LongParticles or FloatParticles
        Input particle values.
    binary: bool
        If `True`, binary erosion and dilation is used. Otherwise ``min`` and ``max`` operators are used. (default: False)
    radius: int
        Radius of the erosion and dilation operations. The 6-connective operations are iterated ``radius`` times. (default: 1)
    inplace: bool
        If `True`, the operation modifies the input ``parts``. Otherwise, a copy of the input is used. (default: False)

    Returns
    -------
    parts_copy: ByteParticles, ShortParticles, LongParticles or FloatParticles
        Output particle values of the same type as the input ``parts``.
    """
    _check_input(apr, parts)
    parts_copy = parts if inplace else parts.copy()
    if binary:
        _internals.binary_erosion(apr, parts_copy, radius)
        _internals.binary_dilation(apr, parts_copy, radius)
    else:
        _internals.erosion(apr, parts_copy, radius)
        _internals.dilation(apr, parts_copy, radius)
    return parts_copy


def closing(apr: APR,
            parts: ParticleData,
            binary: bool = False,
            radius: int = 1,
            inplace: bool = False) -> ParticleData:
    """
    Apply morphological closing (dilation followed by erosion) to an APR image. Only considers face-side neighbors
    (6-connectivity in 3D). If ``radius > 1`` the erosion and dilation are repeated ``radius`` times.

    Parameters
    ----------
    apr: APR
        Input APR data structure.
    parts: ByteParticles, ShortParticles, LongParticles or FloatParticles
        Input particle values.
    binary: bool
        If `True`, binary erosion and dilation is used. Otherwise ``min`` and ``max`` operators are used. (default: False)
    radius: int
        Radius of the erosion and dilation operations. The 6-connective operations are iterated ``radius`` times. (default: 1)
    inplace: bool
        If `True`, the operation modifies the input ``parts``. Otherwise, a copy of the input is used. (default: False)

    Returns
    -------
    parts_copy: ByteParticles, ShortParticles, LongParticles or FloatParticles
        Output particle values of the same type as the input ``parts``.
    """
    _check_input(apr, parts)
    parts_copy = parts if inplace else parts.copy()
    if binary:
        _internals.binary_dilation(apr, parts_copy, radius)
        _internals.binary_erosion(apr, parts_copy, radius)
    else:
        _internals.dilation(apr, parts_copy, radius)
        _internals.erosion(apr, parts_copy, radius)
    return parts_copy


def tophat(apr: APR,
           parts: ParticleData,
           binary: bool = False,
           radius: int = 1) -> ParticleData:
    """
    Apply top-hat transform to an APR image by subtracting an opening of the image.

    Parameters
    ----------
    apr: APR
        Input APR data structure.
    parts: ByteParticles, ShortParticles, LongParticles or FloatParticles
        Input particle values.
    binary: bool
        Use binary opening? (default: False)
    radius: int
        Radius of the opening operation. (default: 1)

    Returns
    -------
    out: ByteParticles, ShortParticles, LongParticles or FloatParticles
        Output particle values of the same type as the input ``parts``.

    See also
    --------
    pyapr.numerics.transform.opening
    """
    _check_input(apr, parts)
    tmp = opening(apr, parts, binary=binary, radius=radius, inplace=False)
    return parts - tmp


def bottomhat(apr: APR,
              parts: ParticleData,
              binary: bool = False,
              radius: int = 1) -> ParticleData:
    """
    Apply bottom-hat transform to an APR image by subtracting it from a closing of itself.

    Parameters
    ----------
    apr: APR
        Input APR data structure.
    parts: ByteParticles, ShortParticles, LongParticles or FloatParticles
        Input particle values.
    binary: bool
        Use binary closing? (default: False)
    radius: int
        Radius of the closing operation. (default: 1)

    Returns
    -------
    out: ByteParticles, ShortParticles, LongParticles or FloatParticles
        Output particle values of the same type as the input ``parts``.

    See also
    --------
    pyapr.numerics.transform.closing
    """
    _check_input(apr, parts)
    tmp = closing(apr, parts, binary=binary, radius=radius, inplace=False)
    return tmp - parts


def remove_small_objects(apr: APR,
                         labels: Union[ByteParticles, ShortParticles, LongParticles],
                         min_volume: int,
                         inplace: bool = False) -> Union[ByteParticles, ShortParticles, LongParticles]:
    """
    Remove objects smaller than a threshold from an input label mask. Assumes that the background value is 0.

    Note: internally allocates a vector of size ``labels.max()+1``. If ``labels`` contains large values, consider
    relabeling consecutively from 0.

    Parameters
    ----------
    apr: APR
        Input APR data structure.
    labels: ByteParticles, ShortParticles or LongParticles
        Input particle label mask.
    min_volume: int
        Remove objects smaller in volume than ``min_volume`` voxels.
    inplace: bool
        If True, the input ``labels`` are modified in-place. Otherwise, a copy is used. (default: False)

    Returns
    -------
    labels_copy: ByteParticles, ShortParticles or LongParticles
        The mask with small objects removed.
    """
    _check_input(apr, labels, __allowed_label_types__)
    labels_copy = labels if inplace else labels.copy()
    _internals.remove_small_objects(apr, labels_copy, int(min_volume))
    return labels_copy


def remove_large_objects(apr: APR,
                         labels: Union[ByteParticles, ShortParticles, LongParticles],
                         max_volume: int,
                         inplace: bool = False) -> Union[ByteParticles, ShortParticles, LongParticles]:
    """
    Remove objects larger than a threshold from an input label mask. Assumes that the background value is 0.

    Note: internally allocates a vector of size ``labels.max()+1``. If ``labels`` contains large values, consider
    relabeling consecutively from 0.

    Parameters
    ----------
    apr: APR
        Input APR data structure.
    labels: ByteParticles, ShortParticles or LongParticles
        Input particle label mask.
    max_volume: int
        Remove objects larger in volume than ``min_volume`` voxels.
    inplace: bool
        If True, the input `labels` are modified in-place. Otherwise, a copy is used. (default: False)

    Returns
    -------
    labels_copy: ByteParticles, ShortParticles or LongParticles
        The mask with large objects removed.
    """
    _check_input(apr, labels, __allowed_label_types__)
    labels_copy = labels if inplace else labels.copy()
    _internals.remove_large_objects(apr, labels_copy, int(max_volume))
    return labels_copy


def remove_small_holes(apr: APR,
                       labels: Union[ByteParticles, ShortParticles, LongParticles],
                       min_volume: int,
                       background_label: int = 0) -> Union[ByteParticles, ShortParticles, LongParticles]:
    """
    Remove holes smaller than a threshold from an input particle mask. Assumes that the background value is 0.

    Parameters
    ----------
    apr: APR
        Input APR data structure.
    labels: ByteParticles, ShortParticles or LongParticles
        Input particle mask.
    min_volume: int
        Remove holes smaller in volume than ``min_volume`` voxels.
    background_label: int
        Value of the background label (default: 0)

    Returns
    -------
    out: ByteParticles, ShortParticles or LongParticles
        The mask with holes removed. If the input mask was binary, a binary mask is returned. Otherwise, connected
        components are recomputed.
    """
    _check_input(apr, labels, __allowed_label_types__)
    mask = (labels == background_label)
    cc_inverted = LongParticles()
    connected_component(apr, mask, cc_inverted)
    _internals.remove_small_objects(apr, cc_inverted, min_volume=int(min_volume))
    mask = (cc_inverted == 0)

    if labels.max() > 1:        # if non-binary input, recompute connected components
        cc = type(labels)()
        connected_component(apr, mask, cc)
        return cc
    return mask


def remove_edge_objects(apr: APR,
                        labels: Union[ByteParticles, ShortParticles, LongParticles],
                        background_label: int = 0,
                        z_edges: bool = True,
                        x_edges: bool = True,
                        y_edges: bool = True,
                        inplace: bool = False) -> Union[ByteParticles, ShortParticles, LongParticles]:
    """
    Remove object labels that intersect with edges of the volume.

    Parameters
    ----------
    apr: APR
        Input APR data structure.
    labels: ByteParticles, ShortParticles or LongParticles
        Input particle label mask.
    background_label: int
        Value of the background label (default: 0)
    z_edges: bool
        If True, checks edges in the z dimension. (default: True)
    x_edges: bool
        If True, checks edges in the x dimension. (default: True)
    y_edges: bool
        If True, checks edges in the y dimension. (default: True)
    inplace: bool
        If True, the input ``labels`` are modified in-place. Otherwise, a copy is used. (default: False)

    Returns
    -------
    labels_copy: ByteParticles, ShortParticles or LongParticles
        The mask with objects on edges set to ``background_label``.
    """
    _check_input(apr, labels, __allowed_label_types__)
    labels_copy = labels if inplace else labels.copy()
    _internals.remove_edge_objects(apr, labels_copy, background_label, z_edges, x_edges, y_edges)
    return labels_copy


def find_perimeter(apr: APR,
                   parts: ParticleData,
                   output: Optional[ParticleData] = None) -> ParticleData:
    """
    Retain the value of particles with at least one zero face-side neighbor, setting remaining particles to 0.
    For example, if the input consists of object labels, the perimeter of each object is returned.

    Parameters
    ----------
    apr: APR
        Input APR data structure.
    parts: ByteParticles, ShortParticles, LongParticles or FloatParticles
        Input particle values.
    output: ByteParticles, ShortParticles, LongParticles or FloatParticles, optional
        Output ParticleData object of the same type as ``parts``. If not provided, or types do not match,
        a new object is generated.

    Returns
    -------
    output: ByteParticles, ShortParticles, LongParticles or FloatParticles
        Particle set with "interior" values set to 0.
    """
    _check_input(apr, parts)
    if not isinstance(output, type(parts)):
        output = type(parts)()
    _internals.find_perimeter(apr, parts, output)
    return output
