import pyapr


def main():

    # Read in an APR
    io_int = pyapr.filegui.InteractiveIO()
    fpath_apr = io_int.get_apr_file_name()  # get APR file path from gui

    # Initialize APRFile for I/O
    aprfile = pyapr.io.APRFile()
    aprfile.set_read_write_tree(True)

    apr = pyapr.APR()
    parts = pyapr.ShortParticles()

    # Read APR and particles from file
    aprfile.open(fpath_apr, 'READ')
    aprfile.read_apr(apr)
    aprfile.read_particles(apr, 'particles', parts)
    aprfile.close()

    # launch the by-slice viewer
    pyapr.viewer.parts_viewer(apr, parts)


if __name__ == '__main__':
    main()

