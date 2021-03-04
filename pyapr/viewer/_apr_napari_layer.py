import pyapr
import numpy as np
from napari.layers import Image
from numbers import Integral


def get_napari_layer(apr: pyapr.APR,
                     parts: (pyapr.ShortParticles, pyapr.FloatParticles),
                     mode: str = 'constant',
                     level_delta: int = 0,
                     name: str = 'APR'):
    """
    Construct a napari 'Image' layer from an APR. Pixel values are reconstructed on the fly via the APRSlicer class.

    Parameters
    ----------
    apr : pyapr.APR
        Input APR data structure
    parts : pyapr.FloatParticles or pyapr.ShortParticles
        Input particle intensities
    mode: str
        Interpolation mode to reconstruct pixel values. Supported values are
            constant:   piecewise constant interpolation
            smooth:     smooth interpolation (via level-adaptive separable smoothing). Note: significantly slower than constant.
            level:      interpolate the particle levels to the pixels
        (default: constant)
    level_delta: int
        Sets the resolution of the reconstruction. The size of the image domain is multiplied by a factor of 2**level_delta.
        Thus, a value of 0 corresponds to the original pixel image resolution, -1 halves the resolution and +1 doubles it.
        (default: 0)
    name: str
        Name of the output Image layer
        (default: 'APR')
    Returns
    -------
    out : napari.layers.Image
        An Image layer of the APR that can be viewed in napari.
    """

    cmin = apr.level_min() if mode == 'level' else parts.min()
    cmax = apr.level_max() if mode == 'level' else parts.max()
    return Image(data=APRSlicer(apr, parts, mode=mode, level_delta=level_delta),
                 rgb=False, multiscale=False, name=name, contrast_limits=[cmin, cmax])


class APRSlicer:
    """
    Helper class allowing (3D) slice indexing. Pixel values in the slice range are reconstructed
    on the fly and returned as an array.
    """
    def __init__(self, apr, parts, mode='constant', level_delta=0):
        self.apr = apr
        self.parts = parts
        self.mode = mode
        self.dims = [np.ceil(x * pow(2, level_delta)) for x in apr.org_dims()]

        if self.mode == 'level':
            self.dtype = np.uint8
        else:
            self.dtype = np.float32 if isinstance(parts, pyapr.FloatParticles) else np.uint16

        self.patch = pyapr.ReconPatch()
        self.patch.level_delta = level_delta
        self.patch.z_end = 1
        self.patch.check_limits(self.apr)

        self.tree_parts = pyapr.FloatParticles()
        pyapr.numerics.fill_tree_mean(self.apr, self.parts, self.tree_parts)

        self._slice = self.new_empty_slice()

        if self.mode == 'constant':
            self.recon = pyapr.numerics.reconstruction.reconstruct_constant
        elif self.mode == 'smooth':
            self.recon = pyapr.numerics.reconstruction.reconstruct_smooth
        elif self.mode == 'level':
            self.recon = pyapr.numerics.reconstruction.reconstruct_level
        else:
            raise ValueError('APRArray mode argument must be \'constant\', \'smooth\' or \'level\'')

    @property
    def shape(self):
        return self.dims[2], self.dims[1], self.dims[0]

    @property
    def ndim(self):
        return 3

    def new_empty_slice(self):
        return np.empty((self.patch.x_end-self.patch.x_begin, self.patch.y_end-self.patch.y_begin), dtype=self.dtype)

    def reconstruct(self):
        if self.mode == 'level':
            self._slice = self.recon(self.apr, patch=self.patch, out_arr=self._slice)
        else:
            self._slice = self.recon(self.apr, self.parts, tree_parts=self.tree_parts, patch=self.patch, out_arr=self._slice)
        return self._slice

    def __getitem__(self, item):
        if isinstance(item, slice):
            self.patch.z_begin = item.start
            self.patch.z_end = item.stop
        elif isinstance(item, tuple):
            limits = [-1, -1, -1, -1, -1, -1]
            for i in range(len(item)):
                if isinstance(item[i], slice):
                    limits[2*i] = item[i].start if item[i].start is not None else -1
                    limits[2*i+1] = item[i].stop if item[i].stop is not None else -1
                elif isinstance(item[i], Integral):
                    limits[2*i] = item[i]
                    limits[2*i+1] = item[i]+1
            self.patch.z_begin, self.patch.z_end, self.patch.x_begin, self.patch.x_end, self.patch.y_begin, self.patch.y_end = limits
        else:
            self.patch.z_begin = item
            self.patch.z_end = item+1
        self.patch.check_limits(self.apr)
        return self.reconstruct()
