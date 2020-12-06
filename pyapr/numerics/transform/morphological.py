import pyapr


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
