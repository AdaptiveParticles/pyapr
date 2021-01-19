import pyapr
import numpy as np


def opening(apr: pyapr.APR,
            parts: (pyapr.ShortParticles, pyapr.FloatParticles),
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
            parts: (pyapr.ShortParticles, pyapr.FloatParticles),
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
           parts: (pyapr.ShortParticles, pyapr.FloatParticles),
           binary: bool = False,
           radius: int = 1):

    # morphological opening
    tmp = opening(apr, parts, binary=binary, radius=radius, inplace=False)

    # return difference
    return parts - tmp


def bottomhat(apr: pyapr.APR,
              parts: (pyapr.ShortParticles, pyapr.FloatParticles),
              binary: bool = False,
              radius: int = 1):

    # morphological closing
    tmp = closing(apr, parts, binary=binary, radius=radius, inplace=False)

    # return difference
    return tmp - parts


def invert(apr: pyapr.APR,
         parts: (pyapr.ShortParticles, pyapr.FloatParticles)):

    mask = pyapr.ShortParticles(apr.total_number_particles())
    for i, elem in enumerate(parts):
        mask[i] = 1 if elem == 0 else 0

    return mask


def remove_small_holes(apr: pyapr.APR,
                       parts: (pyapr.ShortParticles, pyapr.FloatParticles),
                       min_volume: int = 200):

    mask = invert(apr, parts)
    cc_inverted = pyapr.ShortParticles(apr.total_number_particles())
    pyapr.numerics.segmentation.connected_component(apr, mask, cc_inverted)
    pyapr.numerics.transform.remove_small_objects(apr, cc_inverted, min_volume=min_volume)
    mask = invert(apr, cc_inverted)

    if (np.array(parts, copy=True)>1).any():
        # Case where input is a label map
        cc = pyapr.ShortParticles(apr.total_number_particles())
        pyapr.numerics.segmentation.connected_component(apr, mask, cc)
        return cc
    else:
        return mask
