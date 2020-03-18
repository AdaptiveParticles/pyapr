import pyapr
import numpy as np
import argparse


def main(args):
    # Read in an image

    io_int = pyapr.filegui.InteractiveIO()
    fpath_apr = io_int.get_apr_file_name()

    apr = pyapr.APR()

    if args.data_type == 'short':
        parts = pyapr.ShortParticles()
    elif args.data_type == 'float':
        parts = pyapr.FloatParticles()
    else:
        raise Exception('currently the only supported data types are float and short')

    # Initialize APRFile for I/O
    aprfile = pyapr.io.APRFile()
    aprfile.set_read_write_tree(True)

    # Read APR and particles from file
    aprfile.open(fpath_apr, 'READ')
    aprfile.read_apr(apr)
    aprfile.read_particles(apr, 'particles', parts)
    aprfile.close()

    if not isinstance(parts, pyapr.ShortParticles):
        sparts = pyapr.ShortParticles()
        sparts.copy(apr, parts)
        parts = sparts

    pyapr.viewer.parts_viewer(apr, parts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("interactive APR conversion")
    parser.add_argument('--data-type', '-d', type=str, default='short',
                        help='data type of the particles: short or float. default: short')
    args = parser.parse_args()

    main(args)
