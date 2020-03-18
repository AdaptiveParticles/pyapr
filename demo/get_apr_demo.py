import os
import pyapr
import numpy as np
from skimage import io as skio
import argparse


def main(args):

    # Initialize objects
    apr = pyapr.APR()
    par = pyapr.APRParameters()

    if args.data_type == 'short':
        dtype = np.uint16
        parts = pyapr.ShortParticles()
        converter = pyapr.converter.ShortConverter()
    elif args.data_type == 'float':
        dtype = np.float32
        parts = pyapr.FloatParticles()
        converter = pyapr.converter.FloatConverter()
    else:
        raise Exception('currently the only supported data types are float and short')

    # Read in an image (and convert if necessary)
    io_int = pyapr.filegui.InteractiveIO()
    fpath = io_int.get_tiff_file_name()
    img = skio.imread(fpath).astype(dtype)

    # Set some parameters
    par.auto_parameters = False
    par.rel_error = 0.1
    par.gradient_smoothing = 2
    converter.set_parameters(par)
    converter.set_verbose(True)

    # Compute APR and sample particle values

    io_int.interactive_apr(converter, apr, img)

    print("Total number of particles: {} \n".format(apr.total_number_particles()))

    print("Sampling particles ... \n")

    parts.sample_image(apr, img)

    print("Done. \n")

    print("Writing file to disk ... \n")

    fpath_apr = io_int.save_apr_file_name() #get path through gui

    # Initialize APRFile for I/O
    aprfile = pyapr.io.APRFile()
    aprfile.set_read_write_tree(True)

    # Write APR and particles to file
    aprfile.open(fpath_apr, 'WRITE')
    aprfile.write_apr(apr)
    aprfile.write_particles('particles', parts)

    file_sz = aprfile.current_file_size_MB()
    print("APR File Size: {:7.2f} MB \n".format(file_sz))

    mcr = (os.path.getsize(fpath) * 1e-6)/file_sz
    cr = img.size/apr.total_number_particles()

    print("Memory Compression Ratio: {:7.2f}".format(mcr))
    print("Compuational Ratio: {:7.2f}".format(cr))

    aprfile.close()

    print("Done. \n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("interactive APR conversion")
    parser.add_argument('--data-type', '-d', type=str, default='short',
                        help='data type of the particles: short or float. default: short')
    args = parser.parse_args()

    main(args)
