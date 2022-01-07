import pyapr


"""
Read a selected APR from file and display (a rectangular region of) a given z-slice as a point scatter.
"""

# Get APR file path from gui
io_int = pyapr.filegui.InteractiveIO()
fpath_apr = io_int.get_apr_file_name()

# Read APR and particles from file
apr, parts = pyapr.io.read(fpath_apr)

display = True      # display the result as a python plot?
save = False        # save resulting plot as an image?
if save:
    save_path = io_int.save_tiff_file_name()

z = None                       # which slice to display? (default None -> display center slice)
base_markersize = 1
markersize_scale_factor = 2    # markersize = base_markersize * particle_size ** markersize_scale_factor
figsize = None                 # figure size in inches (default None -> determined by xrange, yrange and dpi)
dpi = 50                       # dots per inch (output image dimensions will be dpi*figsize)
xrange = (400, 800)            # range of x values to be plotted
yrange = (400, 800)            # range of y values to be plotted  (if None or out of bounds, the entire range is used)

pyapr.viewer.particle_scatter_plot(apr, parts, z=z, markersize_scale_factor=markersize_scale_factor,
                                   base_markersize=base_markersize, figsize=figsize, dpi=dpi,
                                   save_path=save_path if save else None, xrange=xrange, yrange=yrange,
                                   display=display, cmap='viridis')
