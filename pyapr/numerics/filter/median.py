from _pyaprwrapper.numerics.filter import *
import pyapr

allowed_sizes = [3, 5, 7, 9, 11]


def median_filter_3d(apr, parts, size):
    if size not in allowed_sizes:
        raise ValueError('median_filter_3d received an invalid argument \'size\'. Allowed values are '
                         '3, 5, 7, 9 and 11')

    fname = 'median_filter_{}{}{}'.format(size, size, size)
    output = pyapr.ShortParticles() if isinstance(parts, pyapr.ShortParticles) else pyapr.FloatParticles()
    globals()[fname](apr, parts, output)
    return output


def median_filter_2d(apr, parts, size):
    if size not in allowed_sizes:
        raise ValueError('median_filter_2d received an invalid argument \'size\'. Allowed values are '
                         '3, 5, 7, 9 and 11')

    fname = 'median_filter_1{}{}'.format(size, size)
    output = pyapr.ShortParticles() if isinstance(parts, pyapr.ShortParticles) else pyapr.FloatParticles()
    globals()[fname](apr, parts, output)
    return output
