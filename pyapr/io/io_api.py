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
