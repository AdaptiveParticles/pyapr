import pyapr
from libtiff import TIFFfile
import numpy as np


def read_tiff(filename):
    """
    Read a tiff file into a numpy array
    Usage: zstack = readTiff(inFileName)
    """
    tiff = TIFFfile(filename)
    samples, sample_names = tiff.get_samples()

    out_list = []
    for sample in samples:
        out_list.append(np.copy(sample))

    out = np.concatenate(out_list, axis=-1)

    tiff.close()

    return out


def main():

    io_int = pyapr.filegui.InteractiveIO()

    fpath = io_int.get_tiff_file_name()

    # Read in an image
    img = read_tiff(fpath).astype(np.float32)

    print(fpath)

    # Initialize objects
    apr = pyapr.APR()
    parts = pyapr.FloatParticles()
    par = pyapr.APRParameters()
    converter = pyapr.converter.FloatConverter()

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
    #converter.get_apr_interactive(apr, img)

    io_int.interactive_apr(converter, apr, img)


    #converter.get_apr(apr, img)
    # parts.sample_image(apr, img)
    #
    # # Reconstruct pixel image
    # tmp = pyapr.numerics.reconstruction.recon_pc(apr, parts)
    # recon = np.array(tmp, copy=False)
    #
    # # Compare reconstruction to original
    # print('mean absolute relative error: {}'.format(np.mean(np.abs(img-recon) / img)))


if __name__ == '__main__':

    main()

