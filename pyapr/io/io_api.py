from _pyaprwrapper.io import APRFile
from _pyaprwrapper.data_containers import APR, ByteParticles, ShortParticles, LongParticles, FloatParticles, \
                                          LazyDataByte, LazyDataShort, LazyDataLong, LazyDataFloat
from ..utils import type_to_particles, type_to_lazy_particles
from .._common import _check_input
from typing import Optional, Union, Tuple, List
from warnings import warn


ParticleData = Union[ByteParticles, ShortParticles, LongParticles, FloatParticles]
LazyData = Union[LazyDataByte, LazyDataShort, LazyDataLong, LazyDataFloat]


__all__ = ['read', 'read_apr', 'read_particles', 'read_multichannel',
           'write', 'write_apr', 'write_particles', 'write_multichannel',
           'get_particle_names', 'get_particle_type']


def read(fpath: str,
         apr: Optional[APR] = None,
         parts: Optional[ParticleData] = None,
         t: int = 0,
         channel_name: str = 't',
         parts_name: str = 'particles') -> Tuple[APR, ParticleData]:
    """
    Read APR structure and one set of particles from file.

    Parameters
    ----------
    fpath: str
        APR file path, e.g. `/home/data/test.apr`
    apr: APR, optional
        APR object to read the file into. If None, a new object is generated. (default: None)
    parts: ParticleData, optional
        ParticleData object to read particles into. If provided, the type must match the data in the file.
        If None, a new object of the correct datatype is generated. (default: None)
    t: int
        Time point to read. (default: 0)
    channel_name: str
        Channel to read. (default: `t`)
    parts_name: str
        Name of the particle field to read. (default: `particles`)

    Return
    ------
    apr, parts
        The APR data structure and particle values.

    See also
    --------
    pyapr.io.read_apr, pyapr.io.read_particles, pyapr.io.get_particle_names
    """

    aprfile = APRFile()
    aprfile.open(fpath, 'READ')

    # instantiate output objects if not given
    apr = apr or APR()
    if parts is None:
        dtype = aprfile.get_particle_type(parts_name, apr_or_tree=True, t=t, channel_name=channel_name)
        parts = type_to_particles(dtype)

    # read APR and particle data from file
    aprfile.read_apr(apr, t=t, channel_name=channel_name)
    aprfile.read_particles(apr, parts_name, parts, apr_or_tree=True, t=t, channel_name=channel_name)
    aprfile.close()
    return apr, parts


def write(fpath: str,
          apr: APR,
          parts: ParticleData,
          t: int = 0,
          channel_name: str = 't',
          parts_name: str = 'particles',
          write_linear: bool = True,
          write_tree: bool = True,
          tree_parts: Optional[ParticleData] = None) -> None:
    """
    Write APR structure and particles to file.

    Parameters
    ----------
    fpath: str
        APR file path, e.g. `/home/data/test.apr`
    apr: APR, optional
        APR object to write.
    parts: ParticleData, optional
        ParticleData object to write.
    t: int
        Time point under which the data is written. (default: 0)
    channel_name: str
        Channel under which the data is written. (default: `t`)
    parts_name: str
        Name of the particle field to write. (default: `particles`)
    write_linear: bool
        If `True`, writes linear APR structure, otherwise the sparse (random access) structure is written. The
        linear structure is used in most processing methods, but requires more memory. (default: `True`)
    write_tree: bool
        If `True`, the APR tree structure (all parent nodes of APR particles) is written to file. Results in slightly
        larger files (roughly 14.3% additional particles in 3D), but typically saves time when reading a file and
        subsequently using the tree. (default: `True`)
    tree_parts: ParticleData, optional
        Values of tree particles (computed via e.g. `pyapr.numerics.fill_tree_mean`). If provided, and
        `write_tree=True`, the values are written to file. This allows, e.g. lazy reconstruction at coarse resolutions
        using `pyapr.LazySlicer`. (default: None)

    See also
    --------
    pyapr.io.write_apr, pyapr.io.write_particles
    """

    if not fpath:
        print('Empty path given. Ignoring call to pyapr.io.write')
        return

    _check_input(apr, parts)

    aprfile = APRFile()
    aprfile.set_write_linear_flag(write_linear)
    aprfile.open(fpath, 'WRITE')
    aprfile.write_apr(apr, t=t, channel_name=channel_name, write_tree=write_tree)
    aprfile.write_particles(parts_name, parts, apr_or_tree=True, t=t, channel_name=channel_name)

    if tree_parts is not None and write_tree:
        aprfile.write_particles(parts_name, tree_parts, apr_or_tree=False, t=t, channel_name=channel_name)

    aprfile.close()


def write_particles(fpath: str,
                    parts: ParticleData,
                    t: int = 0,
                    channel_name: str = 't',
                    parts_name: str = 'particles',
                    tree: bool = False,
                    append: bool = True):
    """
    Write particle values to a new or existing file.

    Parameters
    ----------
    fpath: str
        APR file path, e.g. `/home/data/test.apr`
    parts: ParticleData, optional
        ParticleData object to write.
    t: int
        Time point under which the data is written. (default: 0)
    channel_name: str
        Channel under which the data is written. (default: `t`)
    parts_name: str
        Name of the particle field to write. (default: `particles`)
    tree: bool
        If `True`, writes the particles under the tree structure. The provided particles should then correspond
        to the APR tree. Otherwise the data is written under the standard APR structure (default: `False`)
    append: bool
        If `True`, writes the data to an existing file, leaving other fields intact. Otherwise, creates a new
        file, possibly overwriting if the file name already exists. (default: True)

    See also
    --------
    pyapr.io.write, pyapr.io.write_apr
    """

    if not fpath:
        print('Empty path given. Ignoring call to pyapr.io.write_particles')
        return

    assert len(parts) > 0, ValueError(f'Input particle dataset {parts} is empty.')

    aprfile = APRFile()
    aprfile.open(fpath, 'READWRITE' if append else 'WRITE')
    aprfile.write_particles(parts_name, parts, apr_or_tree=(not tree), t=t, channel_name=channel_name)
    aprfile.close()


def read_particles(fpath: str,
                   apr: Optional[APR] = None,
                   parts: Optional[ParticleData] = None,
                   t: int = 0,
                   channel_name: str = 't',
                   parts_name: str = 'particles',
                   tree: bool = False):
    """
    Read particle values from file.

    Parameters
    ----------
    fpath: str
        APR file path, e.g. `/home/data/test.apr`
    apr: APR, optional
        Corresponding APR object, allowing an alternative (equivalent) read method to be used. Currently does not
        affect the result. (default: None)
    parts: ParticleData, optional
        ParticleData object to read particles into. If provided, the type must match the data in the file.
        If None, a new object of the correct datatype is generated. (default: None)
    t: int
        Time point under which the data is written. (default: 0)
    channel_name: str
        Channel under which the data is written. (default: `t`)
    parts_name: str
        Name of the particle field to write. (default: `particles`)
    tree: bool
        If `True`, reads the particles under the tree structure in the file. (default: `False`)

    Return
    ------
    parts: ParticleData
        The particle values.

    See also
    --------
    pyapr.io.read, pyapr.io.read_apr
    """

    aprfile = APRFile()
    aprfile.open(fpath, 'READ')

    if parts is None:
        dtype = aprfile.get_particle_type(parts_name, apr_or_tree=(not tree), t=t, channel_name=channel_name)
        parts = type_to_particles(dtype)

    if apr is not None:
        aprfile.read_particles(apr, parts_name, parts, apr_or_tree=(not tree), t=t, channel_name=channel_name)
    else:
        aprfile.read_particles(parts_name, parts, apr_or_tree=(not tree), t=t, channel_name=channel_name)

    aprfile.close()
    return parts


def write_apr(fpath: str,
              apr: APR,
              t: int = 0,
              channel_name: str = 't',
              write_linear: bool = True,
              write_tree: bool = True,
              append: bool = False):
    """
    Write APR structure to file.

    Parameters
    ----------
    fpath: str
        APR file path, e.g. `/home/data/test.apr`
    apr: APR, optional
        APR object to write.
    t: int
        Time point under which the data is written. (default: 0)
    channel_name: str
        Channel under which the data is written. (default: `t`)
    write_linear: bool
        If `True`, writes linear APR structure, otherwise the sparse (random access) structure is written. The
        linear structure is used in most processing methods, but requires more memory. (default: `True`)
    write_tree: bool
        If `True`, the APR tree structure (all parent nodes of APR particles) is written to file. Results in slightly
        larger files, (roughly 14.3% additional particles in 3D) but typically saves time when reading a file and
        subsequently using the tree. (default: `True`)
    append: bool
        If `True`, writes the data to an existing file, leaving other fields (channels/time points) intact. Otherwise,
        creates a new file, possibly overwriting if the file name already exists. (default: False)

    See also
    --------
    pyapr.io.write, pyapr.io.write_particles
    """

    if not fpath:
        print('Empty path given. Ignoring call to pyapr.io.write_apr')
        return

    assert apr.total_number_particles() > 0, ValueError(f'Input APR {apr} is not initialized.')

    aprfile = APRFile()
    aprfile.set_write_linear_flag(write_linear)
    aprfile.open(fpath, 'READWRITE' if append else 'WRITE')
    aprfile.write_apr(apr, t=t, channel_name=channel_name, write_tree=write_tree)
    aprfile.close()


def read_apr(fpath: str,
             apr: Optional[APR] = None,
             t: int = 0,
             channel_name: str = 't'):
    """
    Read APR structure from file.

    Parameters
    ----------
    fpath: str
        APR file path, e.g. `/home/data/test.apr`
    apr: APR, optional
        APR object to read the file into. If None, a new object is generated. (default: None)
    t: int
        Time point to read. (default: 0)
    channel_name: str
        Channel to read. (default: `t`)

    Return
    ------
    apr: APR
        The APR data structure.

    See also
    --------
    pyapr.io.read, pyapr.io.read_particles
    """

    apr = apr or APR()
    aprfile = APRFile()
    aprfile.open(fpath, 'READ')
    aprfile.read_apr(apr, t=t, channel_name=channel_name)
    aprfile.close()
    return apr


def get_particle_names(fpath: str,
                       t: int = 0,
                       channel_name: str = 't',
                       tree: bool = False) -> List[str]:
    """
    List the particle field names present in an APR file.

    Parameters
    ----------
    fpath: str
        APR file path, e.g. `/home/data/test.apr`
    t: int
        Time point to read. (default: 0)
    channel_name: str
        Channel to read. (default: `t`)
    tree: bool
        If `True`, checks the tree structure under the given channel and time point. (default: False)

    Return
    ------
    names: list of str
        A list containing the detected particle names. If no data is found, returns an empty list.

    See also
    --------
    pyapr.io.read, pyapr.io.read_particles
    """

    aprfile = APRFile()
    aprfile.open(fpath, 'READ')
    names = aprfile.get_particles_names(apr_or_tree=(not tree), t=t, channel_name=channel_name)
    aprfile.close()
    return names


def get_particle_type(fpath: str,
                      t: int = 0,
                      channel_name: str = 't',
                      parts_name: str = 'particles',
                      tree: bool = False):
    """
    Return the datatype of a given particle dataset on file as a string.

    Parameters
    ----------
    fpath: str
        APR file path, e.g. `/home/data/test.apr`
    t: int
        Time point to read. (default: 0)
    channel_name: str
        Channel to read. (default: `t`)
    parts_name: str
        Name of the particle field whose type is to be determined.
    tree: bool
        If `True`, checks the tree structure under the given channel and time point. (default: False)

    Return
    ------
    dtype: str
        String describing the datatype (e.g. `float` or `uint16`)

    See also
    --------
    pyapr.utils.type_to_particles, pyapr.utils.type_to_lazy_particles
    """

    aprfile = APRFile()
    aprfile.open(fpath, 'READ')
    dtype = aprfile.get_particle_type(parts_name, apr_or_tree=(not tree), t=t, channel_name=channel_name)
    aprfile.close()
    return dtype


# TODO: update these
def write_multichannel(fpath, apr, parts_list, t=0, channel_name='t', channel_names_parts=None):

    warn('\'pyapr.io.write_multichannel\' is deprecated and will be removed in a future release. '
         'Use \'pyapr.io.write_apr\' and \'pyapr.io.write_particles\' instead.', DeprecationWarning)

    if not fpath:
        print('Empty path given. Ignoring call to pyapr.io.write')
        return

    if isinstance(parts_list, (tuple, list)):
        for p in parts_list:
            if not isinstance(p, (ShortParticles, FloatParticles)):
                raise AssertionError(
                    'argument \'parts_list\' to pyapr.io.write_multichannel must be a \
                    tuple or list of pyapr.XParticles objects'
                )
    else:
        raise AssertionError(
            'argument \'parts_list\' to pyapr.io.write_multichannel must be a tuple or list of pyapr.XParticles objects'
        )

    if channel_names_parts is None:
        channel_names_parts = ['particles' + str(i) for i in range(len(parts_list))]

    # Initialize APRFile for I/O
    aprfile = APRFile()

    # Write APR and particles to file
    aprfile.open(fpath, 'WRITE')
    aprfile.write_apr(apr, t=t, channel_name=channel_name)
    for i in range(len(parts_list)):
        aprfile.write_particles(channel_names_parts[i], parts_list[i], t=t)
    aprfile.close()


def read_multichannel(fpath, apr, parts_list, t=0, channel_name='t', channel_names_parts=None):

    warn('\'pyapr.io.read_multichannel\' is deprecated and will be removed in a future release. '
         'Use \'pyapr.io.read_apr\' and \'pyapr.io.read_particles\' instead.', DeprecationWarning)

    if isinstance(parts_list, (tuple, list)):
        for p in parts_list:
            if not isinstance(p, (ShortParticles, FloatParticles)):
                raise AssertionError(
                    'argument \'parts_list\' to pyapr.io.read_multichannel must be a \
                    tuple or list of pyapr.XParticles objects')
    else:
        raise AssertionError(
            'argument \'parts_list\' to pyapr.io.read_multichannel must be a tuple or list of pyapr.XParticles objects')

    if channel_names_parts is None:
        channel_names_parts = ['particles' + str(i) for i in range(len(parts_list))]

    # Initialize APRFile for I/O
    aprfile = APRFile()

    # Write APR and particles to file
    aprfile.open(fpath, 'READ')
    aprfile.read_apr(apr, t=t, channel_name=channel_name)
    for i in range(len(parts_list)):
        aprfile.read_particles(apr, channel_names_parts[i], parts_list[i], t=t)
    aprfile.close()
