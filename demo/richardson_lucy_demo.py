import pyapr
import numpy as np


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

    # specify the PSF and number of iterations
    psf = pyapr.numerics.filter.get_gaussian_stencil(5, 0.8, True)
    niter = 10

    fparts = pyapr.FloatParticles()
    fparts.copy(apr, parts)

    # add a small offset to the particle values
    tmp = np.array(fparts, copy=False)
    tmp += 1e-5 * tmp.max()

    pyapr.viewer.parts_viewer(apr, fparts)

    # perform richardson-lucy deconvolution
    output = pyapr.FloatParticles()
    pyapr.numerics.richardson_lucy(apr, fparts, output, psf, niter, use_stencil_downsample=True,
                                   normalize_stencil=True)

    # alternatively, if built with cuda enabled and stencil is of size (3, 3, 3) or (5, 5, 5)
    # pyapr.numerics.richardson_lucy_cuda(apr, fparts, output, psf, niter, use_stencil_downsample=True,
    #                                     normalize_stencil=True)

    # Display the result
    pyapr.viewer.parts_viewer(apr, output)

    # Write the result to file
    fpath_apr = io_int.save_apr_file_name()  # get path through gui

    # Initialize APRFile for I/O
    aprfile = pyapr.io.APRFile()
    aprfile.set_read_write_tree(True)

    # Write APR and particles to file
    aprfile.open(fpath_apr, 'WRITE')
    aprfile.write_apr(apr)
    aprfile.write_particles('particles', output)

    aprfile.close()


if __name__ == '__main__':
    main()
