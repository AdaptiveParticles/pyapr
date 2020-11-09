import pyapr


def main():

    # Get input APR file path from gui
    io_int = pyapr.filegui.InteractiveIO()
    fpath_apr = io_int.get_apr_file_name()

    # Initialize APR and particles objects
    apr = pyapr.APR()
    # parts = pyapr.FloatParticles()
    parts = pyapr.ShortParticles()

    # Read APR and particles from file
    pyapr.io.read(fpath_apr, apr, parts)

    # Raycast viewer currently only works for ShortParticles
    if isinstance(parts, pyapr.FloatParticles):
        tmp = pyapr.ShortParticles()
        tmp.copy(apr, parts)
        parts = tmp

    # Launch the raycast viewer
    pyapr.viewer.raycast_viewer(apr, parts)


if __name__ == '__main__':
    main()
