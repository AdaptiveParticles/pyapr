import pyapr


def read(fpath, apr=None, parts=None, t=0, channel_name='t', parts_name='particles', read_tree=True):

    # Initialize APRFile for I/O
    aprfile = pyapr.io.APRFile()
    aprfile.set_read_write_tree(read_tree)
    aprfile.open(fpath, 'READ')

    # initialize output objects if not given
    apr = apr or pyapr.APR()
    if parts is None:
        dtype = aprfile.get_particle_type(parts_name, apr_or_tree=True, t=t, channel_name=channel_name)
        parts = initialize_particles_type(dtype)

    # read APR and particle data from file
    aprfile.read_apr(apr, t=t, channel_name=channel_name)
    aprfile.read_particles(apr, parts_name, parts, apr_or_tree=True, t=t, channel_name=channel_name)

    aprfile.close()
    return apr, parts


def write(fpath, apr, parts, t=0, channel_name='t', parts_name='particles', write_linear=True, write_tree=True, tree_parts=None):

    if not fpath:
        print('Empty path given. Ignoring call to pyapr.io.write')
        return

    # Initialize APRFile for I/O
    aprfile = pyapr.io.APRFile()
    aprfile.set_read_write_tree(write_tree)
    aprfile.set_write_linear_flag(write_linear)

    # Write APR and particles to file
    aprfile.open(fpath, 'WRITE')
    aprfile.write_apr(apr, t=t, channel_name=channel_name)
    aprfile.write_particles(parts_name, parts, apr_or_tree=True, t=t, channel_name=channel_name)

    if tree_parts is not None and write_tree:
        assert isinstance(tree_parts, (pyapr.ShortParticles, pyapr.FloatParticles))
        aprfile.write_particles(parts_name, tree_parts, apr_or_tree=False, t=t, channel_name=channel_name)

    aprfile.close()


def write_particles(fpath, parts, t=0, channel_name='t', parts_name='particles', tree=False, append=True):
    if not fpath:
        print('Empty path given. Ignoring call to pyapr.io.write_particles')
        return

    aprfile = pyapr.io.APRFile()
    aprfile.open(fpath, 'READWRITE' if append else 'WRITE')
    aprfile.write_particles(parts_name, parts, apr_or_tree=(not tree), t=t, channel_name=channel_name)
    aprfile.close()


def read_particles(fpath, apr=None, parts=None, t=0, channel_name='t', parts_name='particles', tree=False):
    aprfile = pyapr.io.APRFile()
    aprfile.set_read_write_tree(tree)
    aprfile.open(fpath, 'READ')

    if parts is None:
        dtype = aprfile.get_particle_type(parts_name, apr_or_tree=(not tree), t=t, channel_name=channel_name)
        parts = initialize_particles_type(dtype)

    if apr is not None:
        aprfile.read_particles(apr, parts_name, parts, apr_or_tree=(not tree), t=t, channel_name=channel_name)
        aprfile.close()
        return parts

    aprfile.read_particles(parts_name, parts, apr_or_tree=(not tree), t=t, channel_name=channel_name)
    aprfile.close()
    return parts


def write_apr(fpath, apr, t=0, channel_name='t', write_linear=True, write_tree=True):

    if not fpath:
        print('Empty path given. Ignoring call to pyapr.io.write_apr')
        return

    aprfile = pyapr.io.APRFile()
    aprfile.set_read_write_tree(write_tree)
    aprfile.set_write_linear_flag(write_linear)
    aprfile.open(fpath, 'WRITE')
    aprfile.write_apr(apr, t=t, channel_name=channel_name)
    aprfile.close()


def read_apr(fpath, apr=None, t=0, channel_name='t', read_tree=True):
    apr = apr or pyapr.APR()
    aprfile = pyapr.io.APRFile()
    aprfile.set_read_write_tree(read_tree)
    aprfile.open(fpath, 'READ')
    aprfile.read_apr(apr, t=t, channel_name=channel_name)
    aprfile.close()
    return apr


def get_particle_names(fpath, t=0, channel_name='t', tree=False):
    aprfile = pyapr.io.APRFile()
    aprfile.open(fpath, 'READ')
    names = aprfile.get_particles_names(apr_or_tree=(not tree), t=t, channel_name=channel_name)
    aprfile.close()
    return names


def get_particle_type(fpath, t=0, channel_name='t', parts_name='particles', tree=False):
    aprfile = pyapr.io.APRFile()
    aprfile.open(fpath, 'READ')
    dtype = aprfile.get_particle_type(parts_name, apr_or_tree=(not tree), t=t, channel_name=channel_name)
    aprfile.close()
    return dtype


def initialize_particles_type(typestr):
    if typestr == 'uint16':
        return pyapr.ShortParticles()
    if typestr == 'float':
        return pyapr.FloatParticles()
    if typestr == 'uint8':
        return pyapr.ByteParticles()

    print('deduced datatype {} is currently not supported - returning None'.format(typestr))
    return None


# TODO: update these
def write_multichannel(fpath, apr, parts_list, t=0, channel_name='t', channel_names_parts=None):

    if not fpath:
        print('Empty path given. Ignoring call to pyapr.io.write')
        return

    if isinstance(parts_list, (tuple, list)):
        for p in parts_list:
            if not isinstance(p, (pyapr.ShortParticles, pyapr.FloatParticles)):
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
    aprfile = pyapr.io.APRFile()
    aprfile.set_read_write_tree(True)

    # Write APR and particles to file
    aprfile.open(fpath, 'WRITE')
    aprfile.write_apr(apr, t=t, channel_name=channel_name)
    for i in range(len(parts_list)):
        aprfile.write_particles(channel_names_parts[i], parts_list[i], t=t)
    aprfile.close()


def read_multichannel(fpath, apr, parts_list, t=0, channel_name='t', channel_names_parts=None):

    if isinstance(parts_list, (tuple, list)):
        for p in parts_list:
            if not isinstance(p, (pyapr.ShortParticles, pyapr.FloatParticles)):
                raise AssertionError(
                    'argument \'parts_list\' to pyapr.io.read_multichannel must be a \
                    tuple or list of pyapr.XParticles objects')
    else:
        raise AssertionError(
            'argument \'parts_list\' to pyapr.io.read_multichannel must be a tuple or list of pyapr.XParticles objects')

    if channel_names_parts is None:
        channel_names_parts = ['particles' + str(i) for i in range(len(parts_list))]

    # Initialize APRFile for I/O
    aprfile = pyapr.io.APRFile()
    aprfile.set_read_write_tree(True)

    # Write APR and particles to file
    aprfile.open(fpath, 'READ')
    aprfile.read_apr(apr, t=t, channel_name=channel_name)
    for i in range(len(parts_list)):
        aprfile.read_particles(apr, channel_names_parts[i], parts_list[i], t=t)
    aprfile.close()
