import pyapr


def main():

    # Read in an APR
    io_int = pyapr.filegui.InteractiveIO()
    fpath_apr = io_int.get_apr_file_name()  # get APR file path from gui

    apr = pyapr.APR()
    parts = pyapr.ShortParticles()

    # Initialize APRFile for I/O
    aprfile = pyapr.io.APRFile()
    aprfile.set_read_write_tree(True)

    # Read APR and particles from file
    aprfile.open(fpath_apr, 'READ')
    aprfile.read_apr(apr)
    aprfile.read_particles(apr, 'particles', parts)
    aprfile.close()

    pyapr.viewer.raycast_viewer(apr, parts)


if __name__ == '__main__':
    main()
