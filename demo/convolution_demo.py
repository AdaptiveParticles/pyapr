import pyapr
import numpy as np


def main():

    io_int = pyapr.filegui.InteractiveIO()
    fpath_apr = io_int.get_apr_file_name()

    aprfile = pyapr.io.APRFile()
    aprfile.set_read_write_tree(True)

    apr = pyapr.APR()
    parts = pyapr.FloatParticles()

    # Read from APR file
    aprfile.open(fpath_apr, 'READ')
    aprfile.read_apr(apr)
    aprfile.read_particles(apr, 'particles', parts)
    aprfile.close()

    stencil = np.ones((5, 5, 5), dtype=np.float32) / 125

    out = pyapr.FloatParticles()

    pyapr.filter.convolve_cuda(apr, parts, out, stencil, True)

    # viewer only accepts uint16 particles for now...
    sparts = pyapr.ShortParticles()
    sparts.copy(apr, out)

    pyapr.viewer.parts_viewer(apr, sparts)


if __name__ == '__main__':
    main()
