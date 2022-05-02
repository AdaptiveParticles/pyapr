import pyapr


"""
Read a selected APR from file and visualize it via maximum intensity projection.

Scroll to zoom
Click and drag to change the view
"""

# Get input APR file path from gui
io_int = pyapr.utils.InteractiveIO()
fpath_apr = io_int.get_apr_file_name()

# Read APR and particles from file
apr, parts = pyapr.io.read(fpath_apr)

# Launch the raycast viewer
pyapr.viewer.raycast_viewer(apr, parts)
