import pyapr
import numpy as np
import time


def main():

    io_int = pyapr.filegui.InteractiveIO()
    fpath_apr = io_int.get_apr_file_name()  # get APR file path from gui

    aprfile = pyapr.io.APRFile()
    aprfile.set_read_write_tree(True)

    apr = pyapr.APR()
    parts = pyapr.ShortParticles()
    # parts = pyapr.FloatParticles()

    # Read from APR file
    aprfile.open(fpath_apr, 'READ')
    aprfile.read_apr(apr)
    aprfile.read_particles(apr, 'particles', parts)
    aprfile.close()

    # add a small offset to the particle values
    tmp = np.array(parts, copy=False)
    offset = 1e-5 * tmp.max() if isinstance(parts, pyapr.FloatParticles) else 1
    tmp += offset

    # specify the PSF and number of iterations
    psf = np.ones((5, 5, 5), dtype=np.float32) / 125
    niter = 100

    # perform richardson-lucy deconvolution
    out = pyapr.FloatParticles()
    pyapr.numerics.richardson_lucy(apr, parts, out, psf, niter, use_stencil_downsample=True, normalize_stencil=True)

    # Display the result
    pyapr.viewer.parts_viewer(apr, out)

    # Write the result to file
    fpath_apr = io_int.save_apr_file_name()  # get path through gui

    # Initialize APRFile for I/O
    aprfile = pyapr.io.APRFile()
    aprfile.set_read_write_tree(True)

    # Write APR and particles to file
    aprfile.open(fpath_apr, 'WRITE')
    aprfile.write_apr(apr)
    aprfile.write_particles('particles', out)

    aprfile.close()


if __name__ == '__main__':
    main()
