import pyapr
import numpy as np


def opening(apr: pyapr.APR,
            parts: (pyapr.ShortParticles, pyapr.LongParticles, pyapr.FloatParticles),
            binary: bool = False,
            radius: int = 1,
            inplace: bool = False):

    if inplace:
        # reference assignment
        parts_copy = parts
    else:
        # copy input particles
        parts_copy = parts.copy()

    # morphological opening
    pyapr.numerics.transform.erosion(apr, parts_copy, binary=binary, radius=radius)
    pyapr.numerics.transform.dilation(apr, parts_copy, binary=binary, radius=radius)

    return parts_copy


def closing(apr: pyapr.APR,
            parts: (pyapr.ShortParticles, pyapr.LongParticles, pyapr.FloatParticles),
            binary: bool = False,
            radius: int = 1,
            inplace: bool = False):

    if inplace:
        # reference assignment
        parts_copy = parts
    else:
        # copy input particles
        parts_copy = parts.copy()

    # morphological closing
    pyapr.numerics.transform.dilation(apr, parts_copy, binary=binary, radius=radius)
    pyapr.numerics.transform.erosion(apr, parts_copy, binary=binary, radius=radius)

    return parts_copy


def tophat(apr: pyapr.APR,
           parts: (pyapr.ShortParticles, pyapr.LongParticles, pyapr.FloatParticles),
           binary: bool = False,
           radius: int = 1):

    # morphological opening
    tmp = opening(apr, parts, binary=binary, radius=radius, inplace=False)

    # return difference
    return parts - tmp


def bottomhat(apr: pyapr.APR,
              parts: (pyapr.ShortParticles, pyapr.LongParticles, pyapr.FloatParticles),
              binary: bool = False,
              radius: int = 1):

    # morphological closing
    tmp = closing(apr, parts, binary=binary, radius=radius, inplace=False)

    # return difference
    return tmp - parts


def remove_small_holes(apr: pyapr.APR,
                       parts: (pyapr.ShortParticles, pyapr.LongParticles),
                       min_volume: int = 200):

    mask = parts < 1
    cc_inverted = pyapr.ShortParticles()
    pyapr.numerics.segmentation.connected_component(apr, mask, cc_inverted)
    pyapr.numerics.transform.remove_small_objects(apr, cc_inverted, min_volume=min_volume)
    mask = cc_inverted < 1

    if parts.max() > 1:
        # Case where input is a label map
        if isinstance(parts, pyapr.ShortParticles):
            cc = pyapr.ShortParticles()
        elif isinstance(parts, pyapr.LongParticles):
            mask = np.array(mask).astype('uint16')
            mask = pyapr.ShortParticles(mask)
            cc = pyapr.LongParticles()
        pyapr.numerics.segmentation.connected_component(apr, mask, cc)
        return cc
    else:
        return mask


def find_objects(apr: pyapr.APR,
                 labels: (pyapr.ShortParticles, pyapr.LongParticles)):

    max_label = labels.max()
    max_dim = max([apr.org_dims(x) for x in range(3)])
    min_coords = np.full((max_label+1, 3), max_dim+1, dtype=np.int32)
    max_coords = np.zeros((max_label+1, 3), dtype=np.int32)
    pyapr.numerics.transform.find_objects_cpp(apr, labels, min_coords, max_coords)

    max_coords[0, :] = [apr.org_dims(x) for x in [2, 0, 1]]
    min_coords[0, :] = 0

    return min_coords, max_coords


def find_label_centers(apr: pyapr.APR,
                       labels: (pyapr.ShortParticles, pyapr.LongParticles),
                       weights: (None, pyapr.ShortParticles, pyapr.FloatParticles) = None):

    max_label = labels.max()
    coords = np.zeros((max_label+1, 3), dtype=np.float64)
    if weights is not None:
        pyapr.numerics.transform.find_label_centers_weighted_cpp(apr, labels, coords, weights)
    else:
        pyapr.numerics.transform.find_label_centers_cpp(apr, labels, coords)
    return coords[np.any(coords > 0, axis=1), :]


def find_label_volume(apr: pyapr.APR,
                      labels: (pyapr.ShortParticles, pyapr.LongParticles)):

    max_label = labels.max()
    volume = np.zeros((max_label+1), dtype=np.uint64)
    pyapr.numerics.transform.find_label_volume_cpp(apr, labels, volume)

    return volume
