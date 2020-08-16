import pyapr


def main():

    # Get APR file path from gui
    io_int = pyapr.filegui.InteractiveIO()
    fpath_apr = io_int.get_apr_file_name()

    # Initialize APR and particles objects
    apr = pyapr.APR()
    parts = pyapr.ShortParticles()
    # parts = pyapr.FloatParticles()

    # Read APR and particles from file
    pyapr.io.read(fpath_apr, apr, parts)

    # Launch the by-slice viewer
    pyapr.viewer.parts_viewer(apr, parts)


if __name__ == '__main__':
    main()
