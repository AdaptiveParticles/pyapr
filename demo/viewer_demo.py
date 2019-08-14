import pyapr
import numpy as np


def main():
    # Read in an image

    io_int = pyapr.filegui.InteractiveIO()

    apr = pyapr.APR()
    parts = pyapr.ShortParticles()

    # Initialize APRFile for I/O
    aprfile = pyapr.io.APRFile()
    aprfile.set_read_write_tree(True)

    fpath_apr = io_int.get_apr_file_name()

    # Write APR and particles to file
    aprfile.open(fpath_apr, 'READ')
    aprfile.read_apr(apr)
    aprfile.read_particles(apr, 'particles', parts)
    aprfile.close()

    pyapr.viewer.parts_viewer(apr, parts)


if __name__ == '__main__':
    main()