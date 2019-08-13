import pyapr
import numpy as np
import argparse
from skimage import io as skio


def main(args):
    # Read in an image
    fpath = '../LibAPR/test/files/Apr/sphere_120/sphere_original.tif'
    if args.input:
        fpath = args.input

    img = np.array(skio.imread(fpath)).astype(np.uint16)

    # Initialize objects
    apr = pyapr.APR()
    parts = pyapr.ShortParticles()
    par = pyapr.APRParameters()
    converter = pyapr.converter.ShortConverter()

    # Set some parameters
    par.auto_parameters = False
    par.rel_error = 0.1
    par.Ip_th = 0
    par.gradient_smoothing = 3
    par.sigma_th = 20
    par.sigma_th_max = 5
    converter.set_parameters(par)
    converter.set_verbose(True)

    # Compute APR and sample particle values
    converter.get_apr(apr, img)
    parts.sample_image(apr, img)

    pyapr.viewer.parts_viewer(apr, parts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default="")
    args = parser.parse_args()

    main(args)