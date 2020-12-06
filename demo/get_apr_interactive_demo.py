import os
import pyapr
from skimage import io as skio
import numpy as np


def main():
    """
    Interactive APR conversion. Reads in a selected TIFF image for interactive setting of the parameters:
        Ip_th       (background intensity level)
        sigma_th    (local intensity scale threshold)
        grad_th     (gradient threshold)

    Use the sliders to control the adaptation. The red overlay shows (approximately) the regions that will be fully
    resolved (at pixel resolution).

    Once the parameters are set, the final steps of the conversion pipeline are applied to produce the APR and sample
    the particle intensities.

    Note: The effect of grad_th may hide the effect of the other thresholds. It is thus recommended to keep grad_th
    low while setting Ip_th and sigma_th, and then increasing grad_th.
    """

    # Read in an image
    io_int = pyapr.filegui.InteractiveIO()
    fpath = io_int.get_tiff_file_name()  # get image file path from gui (data type must be float32 or uint16)
    img = skio.imread(fpath)

    # Set some parameters (only Ip_th, grad_th and sigma_th are set interactively)
    par = pyapr.APRParameters()
    par.rel_error = 0.1              # relative error threshold
    par.gradient_smoothing = 3       # b-spline smoothing parameter for gradient estimation
    #                                  0 = no smoothing, higher = more smoothing
    par.dx = 1
    par.dy = 1                       # voxel size
    par.dz = 1

    # Compute APR and sample particle values
    apr, parts = pyapr.converter.get_apr_interactive(img, params=par, verbose=True)

    # Display the APR
    pyapr.viewer.parts_viewer(apr, parts)

    # Write the resulting APR to file
    print("Writing APR to file ... \n")
    fpath_apr = io_int.save_apr_file_name()  # get path through gui
    pyapr.io.write(fpath_apr, apr, parts)

    if fpath_apr:
        # Display the size of the file
        file_sz = os.path.getsize(fpath_apr)
        print("APR File Size: {:7.2f} MB \n".format(file_sz * 1e-6))

        # Compute compression ratio
        mcr = os.path.getsize(fpath) / file_sz
        print("Memory Compression Ratio: {:7.2f}".format(mcr))


if __name__ == '__main__':
    main()
