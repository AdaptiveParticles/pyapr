import pyapr


"""
This demo showcases some of the available numerics functionality on a selected APR
"""

io_int = pyapr.utils.InteractiveIO()
fpath_apr = io_int.get_apr_file_name()      # get APR file path from gui

# Read from APR file
apr, parts = pyapr.io.read(fpath_apr)

# Compute gradient along a dimension (Sobel filter). dimension can be 0, 1 or 2
output = pyapr.filter.gradient(apr, parts, dim=0, delta=1.0)
pyapr.viewer.parts_viewer(apr, output)   # Display the result

# Compute gradient magnitude (central finite differences)
par = apr.get_parameters()
output = pyapr.filter.sobel_magnitude(apr, parts, deltas=(par.dy, par.dx, par.dz), output=output)
pyapr.viewer.parts_viewer(apr, output)  # Display the result

# # Compute local standard deviation around each particle
# pyapr.filter.local_std(apr, parts, output, size=(5, 5, 5))
# pyapr.viewer.parts_viewer(apr, output)  # Display the result
