import pyapr


def main():

    io_int = pyapr.filegui.InteractiveIO()
    fpath_apr = io_int.get_apr_file_name()  # get APR file path from gui

    # Initialize APR and particle objects
    apr = pyapr.APR()
    parts = pyapr.ShortParticles()  # input particles can be float32 or uint16
    # parts = pyapr.FloatParticles()

    # Read from APR file
    pyapr.io.read(fpath_apr, apr, parts)

    # Stencil and output must be float32
    stencil = pyapr.numerics.filter.get_gaussian_stencil(size=5, sigma=1, ndims=3, normalize=True)
    out = pyapr.FloatParticles()

    # Convolve using CPU
    pyapr.numerics.convolve(apr, parts, out, stencil, use_stencil_downsample=True,
                            normalize_stencil=True, use_reflective_boundary=False)

    # Alternative convolution methods:
    # pyapr.numerics.convolve_pencil(apr, parts, out, stencil, use_stencil_downsample=True,
    #                                normalize_stencil=True, use_reflective_boundary=False)  # CPU
    # pyapr.numerics.convolve_cuda(apr, parts, out, stencil, use_stencil_downsample=True,
    #                              normalize_stencil=True, use_reflective_boundary=False)    # GPU

    # Display the result
    pyapr.viewer.parts_viewer(apr, out)


if __name__ == '__main__':
    main()
