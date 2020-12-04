import pyapr


def main():
    """
    Read a selected APR from file and visualize it via maximum intensity projection.

    Scroll to zoom
    Click and drag to change the view
    """

    # Get input APR file path from gui
    io_int = pyapr.filegui.InteractiveIO()
    fpath_apr = io_int.get_apr_file_name()

    # Instantiate APR and particles objects
    apr = pyapr.APR()
    parts = pyapr.ShortParticles()
    # parts = pyapr.FloatParticles()

    # Read APR and particles from file
    pyapr.io.read(fpath_apr, apr, parts)

    # Raycast viewer currently only works for ShortParticles
    if isinstance(parts, pyapr.FloatParticles):
        tmp = pyapr.ShortParticles()
        tmp.copy(parts)
        parts = tmp

    # Launch the raycast viewer
    pyapr.viewer.raycast_viewer(apr, parts)


if __name__ == '__main__':
    main()
