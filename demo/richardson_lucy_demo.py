import pyapr


def main():
    """
    Read a selected APR from file and apply Richardson-Lucy deconvolution
    """

    # Get input APR file path from gui
    io_int = pyapr.filegui.InteractiveIO()
    fpath_apr = io_int.get_apr_file_name()

    # Instantiate APR and particles objects
    apr = pyapr.APR()
    parts = pyapr.ShortParticles()
    # parts = pyapr.FloatParticles()

    # Read from APR file
    pyapr.io.read(fpath_apr, apr, parts)

    # Copy particles to float
    fparts = pyapr.FloatParticles()
    fparts.copy(parts)

    # Add a small offset to the particle values to avoid division by 0
    offset = 1e-5 * fparts.max()
    fparts += offset

    # Display the input image
    pyapr.viewer.parts_viewer(apr, fparts)

    # Specify the PSF and number of iterations
    psf = pyapr.numerics.filter.get_gaussian_stencil(size=5, sigma=1, ndims=3, normalize=True)
    niter = 20

    # Perform richardson-lucy deconvolution
    output = pyapr.FloatParticles()
    pyapr.numerics.richardson_lucy(apr, fparts, output, psf, niter, use_stencil_downsample=True,
                                   normalize_stencil=True, resume=False)

    # Alternative using total variation regularization:
    # reg_factor = 1e-2
    # pyapr.numerics.richardson_lucy_tv(apr, fparts, output, psf, niter, reg_factor, use_stencil_downsample=True,
    #                                   normalize_stencil=True, resume=False)

    # Alternatively, if PyLibAPR is built with CUDA enabled and psf is of size (3, 3, 3) or (5, 5, 5)
    # pyapr.numerics.richardson_lucy_cuda(apr, fparts, output, psf, niter, use_stencil_downsample=True,
    #                                     normalize_stencil=True, resume=False)

    # Display the result
    pyapr.viewer.parts_viewer(apr, output)


if __name__ == '__main__':
    main()
