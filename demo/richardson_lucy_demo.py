import pyapr
import numpy as np
from skimage import io as skio
import time
import matplotlib.pyplot as plt
from LR_compare import get_stencil


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

    psf = np.ones((5, 5, 5), dtype=np.float32) / 125

    out = pyapr.FloatParticles()
    pyapr.filter.richardson_lucy(apr, parts, out, psf, 10, True, True)

    # viewer only accepts uint16 particles for now...
    sparts = pyapr.ShortParticles()
    sparts.copy(apr, parts)

    pyapr.viewer.parts_viewer(apr, sparts)


if __name__ == '__main__':
    main()
