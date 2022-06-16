from _pyaprwrapper.data_containers import APR, ReconPatch, ByteParticles, ShortParticles, FloatParticles, LongParticles
from _pyaprwrapper.tree import fill_tree_mean, fill_tree_max
from ..utils import particles_to_type
from .._common import _check_input
from . import reconstruct_constant, reconstruct_level, reconstruct_smooth
import numpy as np
from numbers import Integral
from typing import Union


class APRSlicer:
    """
    Helper class allowing (3D) slice indexing. Pixel values in the slice range are reconstructed
    on the fly and returned as an array.

    Examples
    --------
    >>> import pyapr
    >>> apr, parts = pyapr.io.read('test.apr')
    >>> slicer = pyapr.reconstruction.APRSlicer(apr, parts)
    >>> slicer[15]                     # reconstruct slice at z=15
    >>> slicer[10:15, 42:45, 57:60]    # reconstruct small 3D patch
    """
    def __init__(self,
                 apr: APR,
                 parts: Union[ByteParticles, ShortParticles, LongParticles, FloatParticles],
                 mode: str = 'constant',
                 level_delta: int = 0,
                 tree_mode: str = 'mean'):
        _check_input(apr, parts)
        self.apr = apr
        self.parts = parts
        self.mode = mode
        self.dtype = np.uint8 if self.mode == 'level' else particles_to_type(parts)

        self.patch = ReconPatch()
        self.patch.level_delta = level_delta
        self.patch.z_end = 1
        self.patch.check_limits(self.apr)
        self.dims = []
        self.update_dims()

        self.tree_parts = FloatParticles()

        if tree_mode == 'mean':
            fill_tree_mean(self.apr, self.parts, self.tree_parts)
        elif tree_mode == 'max':
            fill_tree_max(self.apr, self.parts, self.tree_parts)
        else:
            raise ValueError(f'Invalid tree mode {tree_mode}. Allowed values are \'mean\' and \'max\'')

        self._slice = self.new_empty_slice()
        if self.mode == 'constant':
            self.recon = reconstruct_constant
        elif self.mode == 'smooth':
            self.recon = reconstruct_smooth
        elif self.mode == 'level':
            self.recon = reconstruct_level
        else:
            raise ValueError(f'Invalid mode {mode}. Allowed values are \'constant\', \'smooth\' and \'level\'')

    @property
    def shape(self):
        return self.dims[2], self.dims[1], self.dims[0]

    @property
    def ndim(self):
        return 3

    def new_empty_slice(self):
        return np.zeros((self.patch.z_end-self.patch.z_begin,
                         self.patch.x_end-self.patch.x_begin,
                         self.patch.y_end-self.patch.y_begin),
                        dtype=self.dtype)

    def update_dims(self):
        self.dims = [int(np.ceil(self.apr.org_dims(x) * pow(2, self.patch.level_delta))) for x in range(3)]

    def set_level_delta(self, level_delta):
        self.patch.level_delta = level_delta
        self.update_dims()

    def reconstruct(self):
        if self.mode == 'level':
            self._slice = self.recon(self.apr, patch=self.patch, out_arr=self._slice)
        else:
            self._slice = self.recon(self.apr, self.parts, tree_parts=self.tree_parts, patch=self.patch, out_arr=self._slice)
        return self._slice.squeeze()

    def __getitem__(self, item):
        if isinstance(item, slice):
            self.patch.x_begin, self.patch.x_end, self.patch.y_begin, self.patch.y_end = [0, -1, 0, -1]
            self.patch.z_begin = int(item.start) if item.start is not None else -1
            self.patch.z_end = int(item.stop) if item.stop is not None else -1
        elif isinstance(item, tuple):
            limits = [-1, -1, -1, -1, -1, -1]
            for i in range(len(item)):
                if isinstance(item[i], slice):
                    limits[2*i] = int(item[i].start) if item[i].start is not None else -1
                    limits[2*i+1] = int(item[i].stop) if item[i].stop is not None else -1
                elif isinstance(item[i], Integral):
                    limits[2*i] = item[i]
                    limits[2*i+1] = item[i]+1
                elif isinstance(item[i], float):
                    limits[2*i] = int(item[i])
                    limits[2*i+1] = int(item[i]+1)
            self.patch.z_begin, self.patch.z_end, self.patch.x_begin, self.patch.x_end, self.patch.y_begin, self.patch.y_end = limits
        else:
            self.patch.x_begin, self.patch.x_end, self.patch.y_begin, self.patch.y_end = [0, -1, 0, -1]
            self.patch.z_begin = int(item)
            self.patch.z_end = int(item+1)
        self.patch.check_limits(self.apr)
        return self.reconstruct()
