import pyapr


"""
This demo performs graph cut segmentation using maxflow-v3.04 (http://pub.ist.ac.at/~vnk/software.html)
by Yuri Boykov and Vladimir Kolmogorov.

The graph is formed by linking each particle to its face-side neighbours in each dimension.
Terminal edge costs are set based on a smoothed local minimum and the local standard deviation, while
neighbour edge costs are set based on intensity difference, resolution level and local standard deviation.

Note: experimental!
"""

io_int = pyapr.filegui.InteractiveIO()
fpath_apr = io_int.get_apr_file_name()  # get APR file path from gui

# Read from APR file
apr, parts = pyapr.io.read(fpath_apr)

# output must be short (uint16)
mask = pyapr.ShortParticles()

# parameters affecting memory usage
avg_num_neighbours = 3.2            # controls the amount of memory initially allocated for edges.
#                                     if memory is being reallocated, consider increasing this value
z_block_size = 64                   # tiled version only: process chunks of this many z-slices
z_ghost_size = 16                   # tiled version only: use this many "ghost slices" on each side of the chunks

# parameters affecting the edge costs
alpha = 5.0                         # scaling factor for terminal edges
beta = 3.0                          # scaling factor for neighbour edges

# parameters affecting the "local minimum" computation
num_tree_smooth = 3                 # number of "neighbour smoothing" iterations to perform on tree particles
push_depth = 1                      # high-resolution nodes take their values from push_depth levels coarser nodes
num_part_smooth = 3                 # number of "neighbour smoothing" iterations to perform on the APR particles

# additional parameters affecting the costs
intensity_threshold = 0             # lower threshold on absolute intensity
std_window_size = 9                 # size of the window used to compute the local standard deviation
min_var = 30                        # both terminal costs are set to 0 in regions with std lower than this value
num_levels = 2                      # terminal costs are only set for the num_levels finest resolution particles
max_factor = 3.0                    # particles brighter than "local_min + max_factor * local_std" are considered foreground

# Compute graphcut segmentation
pyapr.numerics.segmentation.graphcut(apr, parts, mask, alpha=alpha, beta=beta, avg_num_neighbours=avg_num_neighbours,
                                     num_tree_smooth=num_tree_smooth, num_part_smooth=num_part_smooth, push_depth=push_depth,
                                     intensity_threshold=intensity_threshold, min_var=min_var, std_window_size=std_window_size,
                                     max_factor=max_factor, num_levels=num_levels)

# If you run out of memory, try using this version
# pyapr.numerics.segmentation.graphcut_tiled(apr, parts, mask, alpha=alpha, beta=beta, avg_num_neighbours=avg_num_neighbours,
#                                            z_block_size=z_block_size, z_ghost_size=z_ghost_size, num_tree_smooth=num_tree_smooth,
#                                            num_part_smooth=num_part_smooth, push_depth=push_depth, intensity_threshold=intensity_threshold,
#                                            min_var=min_var, std_window_size=std_window_size,
#                                            max_factor=max_factor, num_levels=num_levels)

# Display the result
pyapr.viewer.parts_viewer(apr, mask)

# Save the result to hdf5
save_path = io_int.save_apr_file_name()
pyapr.io.write(save_path, apr, mask)
