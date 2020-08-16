import pyapr
import numpy as np
from skimage import io as skio


def main():

    # get input APR file path from gui
    io_int = pyapr.filegui.InteractiveIO()
    fpath_apr = io_int.get_apr_file_name()

    # Initialize APR and particle objects
    parts = pyapr.ShortParticles()
    apr = pyapr.APR()

    # Read APR and particles from file
    pyapr.io.read(fpath_apr, apr, parts)

    # Compute piecewise constant, smooth and level reconstructions
    recon = np.array(pyapr.numerics.reconstruction.recon_pc(apr, parts), copy=False)
    smooth_recon = np.array(pyapr.numerics.reconstruction.recon_smooth(apr, parts), copy=False)
    level_recon = np.array(pyapr.numerics.reconstruction.recon_level(apr), copy=False)

    # Save the results
    file_name = fpath_apr.split('/')[-1]
    file_name = file_name.split('.')[:-1]
    file_name = '.'.join(file_name)

    save_path = io_int.save_tiff_file_name(file_name + '_reconstruct_const.tif')
    skio.imsave(save_path, recon)

    save_path = io_int.save_tiff_file_name(file_name + '_reconstruct_smooth.tif')
    skio.imsave(save_path, smooth_recon)

    save_path = io_int.save_tiff_file_name(file_name + '_reconstruct_level.tif')
    skio.imsave(save_path, level_recon)


if __name__ == '__main__':
    main()
