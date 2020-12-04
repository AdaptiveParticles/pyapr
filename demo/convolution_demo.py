import pyapr


def main():
    """
    This demo reads an APR, applies a convolution operation and displays the result
    """

    io_int = pyapr.filegui.InteractiveIO()
    fpath_apr = io_int.get_apr_file_name()  # get APR file path from gui

    # Instantiate APR and particle objects
    apr = pyapr.APR()
    parts = pyapr.ShortParticles()      # input particles can be float32 or uint16
    # parts = pyapr.FloatParticles()

    # Read from APR file
    pyapr.io.read(fpath_apr, apr, parts)

    # Stencil and output must be float32
    stencil = pyapr.numerics.filter.get_gaussian_stencil(size=5, sigma=1, ndims=3, normalize=True)
    out = pyapr.FloatParticles()

    # Convolve using CPU:
    pyapr.numerics.convolve(apr, parts, out, stencil, use_stencil_downsample=True,
                            normalize_stencil=True, use_reflective_boundary=False)

    # Alternative CPU convolution method (produces the same result as 'convolve'):
    # pyapr.numerics.convolve_pencil(apr, parts, out, stencil, use_stencil_downsample=True,
    #                                normalize_stencil=True, use_reflective_boundary=False)

    # Convolve using GPU (stencil must be of shape 3x3x3 or 5x5x5):
    # pyapr.numerics.convolve_cuda(apr, parts, out, stencil, use_stencil_downsample=True,
    #                          normalize_stencil=True, use_reflective_boundary=False)

    # Display the result
    pyapr.viewer.parts_viewer(apr, out)


if __name__ == '__main__':
    main()
