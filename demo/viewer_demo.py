import pyapr
import numpy as np
from demo.io import read_tiff


def main():
    # Read in an image
    fpath = '../LibAPR/test/files/Apr/sphere_120/sphere_original.tif'
    img = read_tiff(fpath).astype(np.uint16)

    # Initialize objects
    apr = pyapr.APR()
    parts = pyapr.ShortParticles()
    par = pyapr.APRParameters()
    converter = pyapr.converter.ShortConverter()

    # Set some parameters
    par.auto_parameters = False
    par.rel_error = 0.1
    par.Ip_th = 0
    par.gradient_smoothing = 2
    par.sigma_th = 50
    par.sigma_th_max = 20
    converter.set_parameters(par)
    converter.set_verbose(True)

    # Compute APR and sample particle values
    converter.get_apr(apr, img)
    parts.sample_image(apr, img)
    apr.init_tree()

    pyapr.viewer.parts_viewer(apr, parts)


if __name__ == '__main__':
    main()