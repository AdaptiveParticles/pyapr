import pyapr


"""
Read a selected APR from file and display it in the z-slice viewer.
"""

# Get APR file path from gui
io_int = pyapr.utils.InteractiveIO()
fpath_apr = io_int.get_apr_file_name()

# Read APR and particles from file
apr, parts = pyapr.io.read(fpath_apr)

# Launch the by-slice viewer
pyapr.viewer.parts_viewer(apr, parts)
