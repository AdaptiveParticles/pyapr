import pyapr
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image


def particle_scatter_plot(apr, parts, z=None, base_markersize=1, markersize_scale_factor=1, save_path=None,
                          figsize=None, dpi=100, xrange=None, yrange=None, display=False, cmap='viridis'):

    if z is None:
        z = apr.z_num(apr.level_max())//2

    arr = np.array(pyapr.viewer.get_points(apr, parts, z), copy=False).squeeze()
    # arr is an array of size (4, num_particles), where the rows are: [x, y, particle size in pixels, intensity]

    # x and y are inverted in the APR
    xsize = apr.y_num(apr.level_max())
    ysize = apr.x_num(apr.level_max())

    if isinstance(xrange, (tuple, list)) and len(xrange) >= 2:
        if 0 <= xrange[0] < xrange[1] < xsize:
            arr = arr[:, xrange[0] <= arr[0]]
            arr = arr[:, arr[0] < xrange[1]]
            xsize = xrange[1] - xrange[0]

    if isinstance(yrange, (tuple, list)) and len(yrange) >= 2:
        if 0 <= yrange[0] < yrange[1] < ysize:
            arr = arr[:, yrange[0] <= arr[1]]
            arr = arr[:, arr[1] < yrange[1]]
            ysize = yrange[1] - yrange[0]

    fig = plt.figure(figsize=figsize if figsize else [xsize/dpi, ysize/dpi], dpi=dpi)
    ax = plt.axes([0, 0, 1, 1], frameon=False, figure=fig)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.invert_yaxis()
    plt.autoscale(tight=True)
    plt.scatter(arr[0], arr[1], s=base_markersize*arr[2]**markersize_scale_factor, c=arr[3], marker='.', cmap=cmap)

    if save_path is not None:
        if save_path.endswith('.tif'):
            png1 = io.BytesIO()
            plt.savefig(png1, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0)
            png2 = Image.open(png1)
            png2.save(save_path)
            png1.close()
        else:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0)

    if display:
        plt.show()
