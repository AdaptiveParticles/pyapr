import os
import pyapr
from skimage import io as skio
import numpy as np


def main():

    # Read in an image
    io_int = pyapr.filegui.InteractiveIO()
    fpath = io_int.get_tiff_file_name()  # get image file path from gui (data type must be float32 or uint16)
    img = skio.imread(fpath)

    while img.ndim < 3:
        img = np.expand_dims(img, axis=0)

    # Initialize APRParameters (only Ip_th, grad_th and sigma_th are set interactively)
    par = pyapr.APRParameters()
    par.auto_parameters = False
    par.rel_error = 0.1
    par.gradient_smoothing = 2
    par.min_signal = 200
    par.noise_sd_estimate = 5

    # Compute APR and sample particle values
    apr, parts = pyapr.converter.get_apr_interactive(img, dtype=img.dtype, params=par, verbose=True)

    # Display the APR
    pyapr.viewer.parts_viewer(apr, parts)

    # Write the resulting APR to file
    print("Writing APR to file ... \n")
    fpath_apr = io_int.save_apr_file_name()  # get path through gui

    # Initialize APRFile for I/O
    aprfile = pyapr.io.APRFile()
    aprfile.set_read_write_tree(True)

    # Write APR and particles to file
    aprfile.open(fpath_apr, 'WRITE')
    aprfile.write_apr(apr)
    aprfile.write_particles('particles', parts)

    # Compute compression and computational ratios
    file_sz = aprfile.current_file_size_MB()
    print("APR File Size: {:7.2f} MB \n".format(file_sz))

    mcr = os.path.getsize(fpath) * 1e-6 / file_sz
    cr = img.size/apr.total_number_particles()

    print("Memory Compression Ratio: {:7.2f}".format(mcr))
    print("Compuational Ratio: {:7.2f}".format(cr))

    aprfile.close()

    print("Done. \n")


if __name__ == '__main__':
    main()
