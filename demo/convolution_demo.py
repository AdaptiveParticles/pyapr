import pyapr
import numpy as np


def main():

    io_int = pyapr.filegui.InteractiveIO()
    fpath_apr = io_int.get_apr_file_name()  # get APR file path from gui

    aprfile = pyapr.io.APRFile()
    aprfile.set_read_write_tree(True)

    # Initialize APR and particle objects
    # parts = pyapr.ShortParticles()  # input particles can be float32 or uint16
    parts = pyapr.FloatParticles()
    apr = pyapr.APR()

    # Read from APR file
    aprfile.open(fpath_apr, 'READ')
    aprfile.read_apr(apr)
    aprfile.read_particles(apr, 'particles', parts)
    aprfile.close()

    # stencil and output must be float32
    stencil = np.ones((5, 5, 5), dtype=np.float32) / 125

    # convolve using cpu
    out = pyapr.FloatParticles()
    pyapr.numerics.convolve(apr, parts, out, stencil, use_stencil_downsample=True, normalize_stencil=True, use_reflective_boundary=False)

    # alternative convolution methods
    # pyapr.numerics.convolve_pencil(apr, parts, out, stencil, use_stencil_downsample=True, normalize_stencil=True, use_reflective_boundary=False)
    # pyapr.numerics.convolve_cuda(apr, parts, out, stencil, use_stencil_downsample=True, normalize_stencil=True, use_reflective_boundary=False)

    # display the result
    pyapr.viewer.parts_viewer(apr, out)


if __name__ == '__main__':
    main()
