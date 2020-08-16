import pyapr
import numpy as np


def main():

    # Get input APR file path from gui
    io_int = pyapr.filegui.InteractiveIO()
    fpath_apr = io_int.get_apr_file_name()

    # Initialize APR and particles objects
    apr = pyapr.APR()
    parts = pyapr.ShortParticles()
    # parts = pyapr.FloatParticles()

    # Read from APR file
    pyapr.io.read(fpath_apr, apr, parts)

    # Specify the PSF and number of iterations
    psf = pyapr.numerics.filter.get_gaussian_stencil(size=5, sigma=0.8, ndims=3, normalize=True)
    niter = 10

    # Copy particles to float
    fparts = pyapr.FloatParticles()
    fparts.copy(apr, parts)

    # Add a small offset to the particle values to avoid division by 0
    tmp = np.array(fparts, copy=False)
    tmp += 1e-5 * tmp.max()

    # Display the input image
    pyapr.viewer.parts_viewer(apr, fparts)

    # Perform richardson-lucy deconvolution
    output = pyapr.FloatParticles()
    pyapr.numerics.richardson_lucy(apr, fparts, output, psf, niter, use_stencil_downsample=True, normalize_stencil=True)

    # Alternatively, if PyLibAPR is built with cuda enabled and stencil is of size (3, 3, 3) or (5, 5, 5)
    # pyapr.numerics.richardson_lucy_cuda(apr, fparts, output, psf, niter, use_stencil_downsample=True, normalize_stencil=True)

    # Display the result
    pyapr.viewer.parts_viewer(apr, output)

    # Write the result to file
    fpath_apr_save = io_int.save_apr_file_name()  # get path through gui
    pyapr.io.write(fpath_apr_save, apr, output)


if __name__ == '__main__':
    main()
