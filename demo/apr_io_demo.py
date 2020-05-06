import pyapr
import numpy as np
from skimage import io as skio


def main():

    # Read in an image
    io_int = pyapr.filegui.InteractiveIO()
    fpath = io_int.get_tiff_file_name()
    img = skio.imread(fpath).astype(np.uint16)

    # Initialize objects
    apr = pyapr.APR()
    parts = pyapr.ShortParticles()
    par = pyapr.APRParameters()
    converter = pyapr.converter.ShortConverter()

    # Set some parameters
    par.auto_parameters = False
    par.rel_error = 0.1
    par.Ip_th = 10
    par.grad_th = 50
    par.gradient_smoothing = 2
    par.sigma_th = 100
    par.sigma_th_max = 50
    converter.set_parameters(par)
    converter.set_verbose(False)

    # Compute APR and sample particle values
    converter.get_apr(apr, img)

    # Compute and display the computational ratio
    numParts = apr.total_number_particles()
    numPix = img.size
    CR = numPix / numParts

    print('input image size: {} pixels, APR size: {} particles --> Computational Ratio: {}'.format(numPix, numParts, CR))

    # Sample particle intensities
    parts.sample_image(apr, img)

    # Save the APR to file
    fpath_apr = io_int.save_apr_file_name()

    # Initialize APRFile for I/O
    aprfile = pyapr.io.APRFile()
    aprfile.set_read_write_tree(True)

    # Write APR and particles to file
    aprfile.open(fpath_apr, 'WRITE')
    aprfile.write_apr(apr)
    aprfile.write_particles('particles', parts)
    aprfile.close()

    # Read the newly written file

    # Initialize objects for reading in data
    apr2 = pyapr.APR()
    parts2 = pyapr.ShortParticles()

    # Read from APR file
    aprfile.open(fpath_apr, 'READ')
    aprfile.read_apr(apr2)
    aprfile.read_particles(apr2, 'particles', parts2)
    aprfile.close()

    # Reconstruct pixel image
    tmp = pyapr.numerics.reconstruction.recon_pc(apr, parts)
    recon = np.array(tmp, copy=False)


if __name__ == '__main__':
    main()
