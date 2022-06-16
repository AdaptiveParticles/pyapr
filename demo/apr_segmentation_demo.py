import pyapr


"""
This demo performs graph cut segmentation using maxflow-v3.04 (http://pub.ist.ac.at/~vnk/software.html)
by Yuri Boykov and Vladimir Kolmogorov.

The graph is formed by linking each particle to its face-side neighbours in each dimension.
Terminal edge costs are set based on a smoothed local minimum and the local standard deviation, while
neighbour edge costs are set based on intensity difference, resolution level and local standard deviation.

Note: experimental!
"""

io_int = pyapr.utils.InteractiveIO()
fpath_apr = io_int.get_apr_file_name()  # get APR file path from gui

# Read from APR file
apr, parts = pyapr.io.read(fpath_apr)

# Compute graphcut segmentation (note that changing the parameters may greatly affect the result)
mask = pyapr.segmentation.graphcut(apr, parts, intensity_threshold=100, min_std=10, num_levels=3)

# Display the result
pyapr.viewer.parts_viewer(apr, mask)
