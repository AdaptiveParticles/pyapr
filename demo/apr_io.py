import pyapr
import numpy as np
from skimage import io as skio
import time


def main():

    # Read in an image
    io_int = pyapr.filegui.InteractiveIO()
    fpath = io_int.get_tiff_file_name()

    start = time.time()
    img = skio.imread(fpath).astype(np.float32)
    print('read image took: {} seconds'.format(time.time()-start))

    # Initialize objects
    apr = pyapr.APR()
    parts = pyapr.FloatParticles()
    par = pyapr.APRParameters()
    converter = pyapr.converter.FloatConverter()

    # Set some parameters
    par.auto_parameters = False
    par.rel_error = 0.1
    par.Ip_th = 100
    par.grad_th = 70
    par.gradient_smoothing = 2
    par.sigma_th = 100
    par.sigma_th_max = 50
    converter.set_parameters(par)
    converter.set_verbose(True)

    # Compute APR and sample particle values
    start = time.time()
    converter.get_apr(apr, img)
    print('get apr took: {} seconds'.format(time.time()-start))

    numParts = apr.total_number_particles()
    numPix = img.size
    CR = numPix / numParts

    print('input image size: {} pixels, APR size: {} particles --> Computational Ratio: {}'.format(numPix, numParts, CR))

    start = time.time()
    parts.sample_image(apr, img)
    print('sample image took: {} seconds'.format(time.time()-start))

    fpath_apr = io_int.save_apr_file_name()

    start = time.time()

    # Initialize APRFile for I/O
    aprfile = pyapr.io.APRFile()
    aprfile.set_read_write_tree(True)

    # Write APR and particles to file
    aprfile.open(fpath_apr, 'WRITE')
    aprfile.write_apr(apr)
    aprfile.write_particles('particles', parts)
    aprfile.close()

    print('write apr took: {} seconds'.format(time.time()-start))

    start = time.time()
    # Initialize objects for reading in data
    apr2 = pyapr.APR()
    parts2 = pyapr.FloatParticles()

    # Read from APR file
    aprfile.open(fpath_apr, 'READ')
    aprfile.read_apr(apr2)
    aprfile.read_particles(apr2, 'particles', parts2)
    aprfile.close()

    print('read apr took: {} seconds'.format(time.time()-start))

    start = time.time()
    # Reconstruct pixel image
    tmp = pyapr.numerics.reconstruction.recon_pc(apr, parts)
    recon = np.array(tmp, copy=False)
    print('reconstruct image took: {} seconds'.format(time.time()-start))

    # Compare reconstructed image to original
    # print('mean absolute relative error: {}'.format(np.mean(np.abs(img-recon) / img)))


if __name__ == '__main__':
    main()


