from _pyaprwrapper.data_containers import APR, ByteParticles, ShortParticles, FloatParticles, LongParticles
from _pyaprwrapper.viewer import get_points
from .._common import _check_input
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
from typing import Union, Optional, Tuple, List, Any


def particle_scatter_plot(apr: APR,
                          parts: Union[ByteParticles, ShortParticles, FloatParticles, LongParticles],
                          z: Optional[int] = None,
                          base_markersize: int = 1,
                          markersize_scale_factor: int = 1,
                          save_path: Optional[str] = None,
                          figsize: Optional[Any] = None,
                          dpi: int = 100,
                          xrange: Optional[Union[Tuple, List]] = None,
                          yrange: Optional[Union[Tuple, List]] = None,
                          display: bool = False,
                          cmap: str = 'viridis'):
    """
    Plot particles in a z-slice (sub-) region as dots colored by intensity and sized according to particle size.
    Uses matplotlib for plotting.

    Parameters
    ----------
    apr: APR
        Input APR data structure.
    parts: ParticleData
        Input particle intensity values
    z: int, optional
        Index of the z-slice to plot. If `None`, the center slice of the volume is taken. (default: None)
    base_markersize: int
        Marker size of the finest dots to plot.
    markersize_scale_factor: int
        Grow dot size exponentially according to `base_markersize * particle_side_length ** markersize_scale_factor`.
    save_path: str, optional
        If provided, the resulting figure is saved to this path.
    figsize: Any, optional
        Size specification of the matplotlib window.
    dpi: int
        Figure resolution in dots-per-inch.
    xrange: tuple or list, optional
        Specify the range to plot in the x dimension. If `None`, plots the entire image range. (default: None)
    yrange: tuple or list, optional
        Specify the range to plot in the y dimension. If `None`, plots the entire image range. (default: None)
    display: bool
        If `True`, calls matplotlib.pyplot.show() to display the figure.
    cmap: str
        Matplotlib color map to use.
    """
    _check_input(apr, parts)
    if z is None:
        z = apr.z_num(apr.level_max())//2

    arr = np.array(get_points(apr, parts, z), copy=False).squeeze()
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
