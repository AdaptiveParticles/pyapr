import pyapr
from libtiff import TIFFfile
import numpy as np
from skimage import io as skio


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
    #fpath = "/Users/cheesema/PhD/PostDoc/PyAPR/demo_files/sphere_long.tif"

    # Read in an image
    img = skio.imread(fpath)

    print(fpath)

    # Initialize objects
    apr = pyapr.APR()
    parts = pyapr.ShortParticles()
    par = pyapr.APRParameters()
    converter = pyapr.converter.ShortConverter()

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

    mcr = (img.size*2*pow(10, -6))/file_sz
    cr = img.size/apr.total_number_particles()

    print("Memory Compression Ratio: {:7.2f}".format(mcr))
    print("Compuational Ratio: {:7.2f}".format(cr))

    aprfile.close()

    print("Done. \n")

    pyapr.viewer.parts_viewer(apr, parts)



if __name__ == '__main__':

    main()

