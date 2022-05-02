from _pyaprwrapper.data_containers import LazyAccess, LazyIterator, ReconPatch
from ..io import APRFile, get_particle_type, get_particle_names, initialize_lazy_particles_type
from .reconstruct import reconstruct_constant_lazy, reconstruct_level_lazy, reconstruct_smooth_lazy
import numpy as np
from numbers import Integral


class LazySlicer:
    """
    Helper class allowing (3D) slice indexing. Pixel values in the slice range are reconstructed lazily (from file)
    on the fly and returned as an array.

    Note
    ----
    Requires the tree structure and corresponding particle values to be present in the file. This can,
    for example, be achieved as follows:

    >>> import pyapr
    >>> apr, parts = pyapr.io.read('file_without_tree.apr')
    >>> tree_parts = type(parts)()                                # new ParticleData object of same type as parts
    >>> pyapr.numerics.fill_tree_mean(apr, parts, tree_parts)
    >>> pyapr.io.write('file_with_tree.apr', apr, parts, write_tree=True, tree_parts=tree_parts)
    """
    def __init__(self, file_path: str, level_delta: int = 0, mode: str = 'constant'):

        self.mode = mode

        self.path = file_path
        self.aprfile = APRFile()
        self.aprfile.set_write_linear_flag(True)
        self.aprfile.open(self.path, "READ")

        # initialize lazy APR access
        self.apr_access = LazyAccess()
        self.apr_access.init(self.aprfile)
        self.apr_access.open()
        self.apr_it = LazyIterator(self.apr_access)

        # initialize lazy tree access
        self.tree_access = LazyAccess()
        self.tree_access.init_tree(self.aprfile)
        self.tree_access.open()
        self.tree_it = LazyIterator(self.tree_access)

        # initialize lazy particle data
        parts_name = get_particle_names(self.path, tree=False)
        parts_type = get_particle_type(self.path, parts_name=parts_name[0], tree=False)
        self.parts = initialize_lazy_particles_type(parts_type)
        self.parts.init(self.aprfile, parts_name[0], 0, 't')
        self.parts.open()

        self.dtype = parts_type

        # initialize lazy tree data
        tree_parts_name = get_particle_names(self.path, tree=True)
        tree_parts_type = get_particle_type(self.path, parts_name=tree_parts_name[0], tree=True)
        self.tree_parts = initialize_lazy_particles_type(tree_parts_type)
        self.tree_parts.init_tree(self.aprfile, tree_parts_name[0], 0, 't')
        self.tree_parts.open()

        self.patch = ReconPatch()
        self.patch.level_delta = level_delta
        self.patch.z_end = 1
        self.patch.check_limits(self.apr_access)
        self.dims = []
        self.update_dims()

        self._slice = self.new_empty_slice()

        if self.mode == 'constant':
            self.recon = reconstruct_constant_lazy
        elif self.mode == 'smooth':
            self.recon = reconstruct_smooth_lazy
        elif self.mode == 'level':
            self.recon = reconstruct_level_lazy
        else:
            raise ValueError('APRArray mode argument must be \'constant\', \'smooth\' or \'level\'')

        self.order = [2, 1, 0]

    def transpose(self, order):
        print(order)
        self.order = [self.order[i] for i in order]

    @property
    def shape(self):
        return self.dims[2], self.dims[1], self.dims[0]

    @property
    def ndim(self):
        return 3

    def new_empty_slice(self):
        return np.zeros((self.patch.z_end-self.patch.z_begin, self.patch.x_end-self.patch.x_begin, self.patch.y_end-self.patch.y_begin), dtype=self.dtype)

    def update_dims(self):
        self.dims = [int(np.ceil(self.apr_access.org_dims(x) * pow(2, self.patch.level_delta))) for x in range(3)]

    def set_level_delta(self, level_delta):
        self.patch.level_delta = level_delta
        self.update_dims()

    def reconstruct(self):
        if self.mode == 'level':
            self._slice = self.recon(self.apr_it, self.tree_it, self.patch, out_arr=self._slice)
        else:
            self._slice = self.recon(self.apr_it, self.tree_it, self.parts,
                                     self.tree_parts, self.patch, out_arr=self._slice)
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
        self.patch.check_limits(self.apr_access)
        return self.reconstruct()
