import pyapr


def read(fpath, apr, parts, t=0, channel_name_apr='t', channel_name_parts='particles'):

    # Initialize APRFile for I/O
    aprfile = pyapr.io.APRFile()
    aprfile.set_read_write_tree(True)

    # Read APR and particles from file
    aprfile.open(fpath, 'READ')
    aprfile.read_apr(apr, t=t, channel_name=channel_name_apr)
    aprfile.read_particles(apr, channel_name_parts, parts)
    aprfile.close()


def write(fpath, apr, parts, t=0, channel_name_apr='t', channel_name_parts='particles'):

    # Initialize APRFile for I/O
    aprfile = pyapr.io.APRFile()
    aprfile.set_read_write_tree(True)

    # Write APR and particles to file
    aprfile.open(fpath, 'WRITE')
    aprfile.write_apr(apr, t=t, channel_name=channel_name_apr)
    aprfile.write_particles(channel_name_parts, parts, t=t)
    aprfile.close()


def write_multichannel(fpath, apr, parts_list, t=0, channel_name_apr='t', channel_names_parts=None):

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
    aprfile.write_apr(apr, t=t, channel_name=channel_name_apr)
    for i in range(len(parts_list)):
        aprfile.write_particles(channel_names_parts[i], parts_list[i], t=t)
    aprfile.close()


def read_multichannel(fpath, apr, parts_list, t=0, channel_name_apr='t', channel_names_parts=None):

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
    aprfile.read_apr(apr, t=t, channel_name=channel_name_apr)
    for i in range(len(parts_list)):
        aprfile.read_particles(apr, channel_names_parts[i], parts_list[i], t=t)
    aprfile.close()
