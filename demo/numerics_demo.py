import pyapr


"""
This demo showcases some of the available numerics functionality on a selected APR
"""

io_int = pyapr.utils.InteractiveIO()
fpath_apr = io_int.get_apr_file_name()      # get APR file path from gui

# Read from APR file
apr, parts = pyapr.io.read(fpath_apr)

output = pyapr.FloatParticles()

# Compute gradient along a dimension (Sobel filter). dimension can be 0, 1 or 2
pyapr.filter.gradient_sobel(apr, parts, output, dim=0, delta=1.0)
pyapr.viewer.parts_viewer(apr, output)   # Display the result

# Compute gradient magnitude (central finite differences)
par = apr.get_parameters()
pyapr.filter.gradient_magnitude_cfd(apr, parts, output, deltas=(par.dy, par.dx, par.dz))
pyapr.viewer.parts_viewer(apr, output)  # Display the result

# Compute local standard deviation around each particle
pyapr.filter.local_std(apr, parts, output, size=(5, 5, 5))
pyapr.viewer.parts_viewer(apr, output)  # Display the result
