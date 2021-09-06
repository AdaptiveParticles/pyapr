import pyapr
import numpy as np
from numbers import Integral


class LazySlicer:
    """
    Helper class allowing (3D) slice indexing. Pixel values in the slice range are reconstructed
    on the fly and returned as an array.
    """
    def __init__(self, file_path, level_delta=0):
        self.path = file_path
        self.aprfile = pyapr.io.APRFile()
        self.aprfile.set_read_write_tree(True)
        self.aprfile.set_write_linear_flag(True)
        self.aprfile.open(self.path, "READ")

        # initialize lazy APR access
        self.apr_access = pyapr.LazyAccess()
        self.apr_access.init(self.aprfile)
        self.apr_access.open()
        self.apr_it = pyapr.LazyIterator(self.apr_access)

        # initialize lazy tree access
        self.tree_access = pyapr.LazyAccess()
        self.tree_access.init_tree(self.aprfile)
        self.tree_access.open()
        self.tree_it = pyapr.LazyIterator(self.tree_access)

        # initialize lazy particle data
        parts_name = pyapr.io.get_particle_names(self.path, tree=False)
        parts_type = pyapr.io.get_particle_type(self.path, parts_name=parts_name[0], tree=False)
        self.parts = pyapr.io.initialize_lazy_particles_type(parts_type)
        self.parts.init_file(self.aprfile, parts_name[0], True)
        self.parts.open()

        self.dtype = parts_type

        # initialize lazy tree data
        tree_parts_name = pyapr.io.get_particle_names(self.path, tree=True)
        tree_parts_type = pyapr.io.get_particle_type(self.path, parts_name=tree_parts_name[0], tree=True)
        self.tree_parts = pyapr.io.initialize_lazy_particles_type(tree_parts_type)
        self.tree_parts.init_file(self.aprfile, tree_parts_name[0], False)
        self.tree_parts.open()

        self.patch = pyapr.ReconPatch()
        self.patch.level_delta = level_delta
        self.patch.z_end = 1
        self.patch.check_limits(self.apr_access)
        self.dims = []
        self.update_dims()

        self._slice = self.new_empty_slice()

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
        self.dims = [np.ceil(self.apr_access.org_dims(x) * pow(2, self.patch.level_delta)) for x in range(3)]

    def set_level_delta(self, level_delta):
        self.patch.level_delta = level_delta
        self.update_dims()

    def reconstruct(self):
        self._slice = pyapr.numerics.reconstruction.reconstruct_lazy(self.apr_it, self.tree_it, self.parts,
                                                                     self.tree_parts, self.patch, out_arr=self._slice)
        return self._slice.squeeze()

    def __getitem__(self, item):
        if isinstance(item, slice):
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
            self.patch.z_begin = int(item)
            self.patch.z_end = int(item+1)
        self.patch.check_limits(self.apr_access)
        return self.reconstruct()
