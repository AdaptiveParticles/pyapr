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

    parts_copy = parts if inplace else parts.copy()

    # morphological opening
    erosion(apr, parts_copy, binary=binary, radius=radius)
    dilation(apr, parts_copy, binary=binary, radius=radius)

    return parts_copy


def closing(apr: APR,
            parts: Union[ShortParticles, LongParticles, FloatParticles],
            binary: bool = False,
            radius: int = 1,
            inplace: bool = False) -> Union[ShortParticles, LongParticles, FloatParticles]:

    parts_copy = parts if inplace else parts.copy()

    # morphological closing
    dilation(apr, parts_copy, binary=binary, radius=radius)
    erosion(apr, parts_copy, binary=binary, radius=radius)

    return parts_copy


def tophat(apr: APR,
           parts: Union[ShortParticles, LongParticles, FloatParticles],
           binary: bool = False,
           radius: int = 1) -> Union[ShortParticles, LongParticles, FloatParticles]:

    tmp = opening(apr, parts, binary=binary, radius=radius, inplace=False)
    return parts - tmp


def bottomhat(apr: APR,
              parts: Union[ShortParticles, LongParticles, FloatParticles],
              binary: bool = False,
              radius: int = 1) -> Union[ShortParticles, LongParticles, FloatParticles]:

    # morphological closing
    tmp = closing(apr, parts, binary=binary, radius=radius, inplace=False)

    # return difference
    return tmp - parts


def remove_small_holes(apr: APR,
                       parts: Union[ShortParticles, LongParticles],
                       min_volume: int = 200) -> Union[ShortParticles, LongParticles]:

    mask = parts < 1
    cc_inverted = ShortParticles()
    connected_component(apr, mask, cc_inverted)
    remove_small_objects(apr, cc_inverted, min_volume=min_volume)
    mask = cc_inverted < 1

    if parts.max() > 1:
        # Case where input is a label map
        if isinstance(parts, ShortParticles):
            cc = ShortParticles()
        elif isinstance(parts, LongParticles):
            mask = np.array(mask).astype('uint16')
            mask = ShortParticles(mask)
            cc = LongParticles()
        connected_component(apr, mask, cc)
        return cc
    else:
        return mask


def find_objects(apr: APR,
                 labels: Union[ShortParticles, LongParticles]) -> Tuple[np.ndarray, np.ndarray]:

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

    max_label = labels.max()
    coords = np.zeros((max_label+1, 3), dtype=np.float64)
    if weights is not None:
        find_label_centers_weighted_cpp(apr, labels, coords, weights)
    else:
        find_label_centers_cpp(apr, labels, coords)
    return coords[np.any(coords > 0, axis=1), :]


def find_label_volume(apr: APR,
                      labels: Union[ShortParticles, LongParticles]) -> np.ndarray:

    max_label = labels.max()
    volume = np.zeros((max_label+1), dtype=np.uint64)
    find_label_volume_cpp(apr, labels, volume)

    return volume
