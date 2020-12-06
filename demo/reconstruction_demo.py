import pyapr
import numpy as np
from skimage import io as skio


def main():
    """
    This demo illustrates three different pixel image reconstruction methods:

        piecewise constant      each pixel takes the value of the particle whose cell contains the pixel
        smooth                  additionally smooths regions of coarser resolution to reduce 'blockiness'
        level                   each pixel takes the value of the resolution level of the particle cell it belongs to
    """

    # get input APR file path from gui
    io_int = pyapr.filegui.InteractiveIO()
    fpath_apr = io_int.get_apr_file_name()

    # Instantiate APR and particle objects
    parts = pyapr.ShortParticles()
    apr = pyapr.APR()

    # Read APR and particles from file
    pyapr.io.read(fpath_apr, apr, parts)

    # Compute piecewise constant reconstruction
    pc_recon = np.array(pyapr.numerics.reconstruction.recon_pc(apr, parts), copy=False)

    # Compute smooth reconstruction
    smooth_recon = np.array(pyapr.numerics.reconstruction.recon_smooth(apr, parts), copy=False)

    # Compute level reconstruction
    level_recon = np.array(pyapr.numerics.reconstruction.recon_level(apr), copy=False)

    # Save the results
    file_name = fpath_apr.split('/')[-1]
    file_name = file_name.split('.')[:-1]
    file_name = '.'.join(file_name)

    save_path = io_int.save_tiff_file_name(file_name + '_reconstruct_const.tif')
    skio.imsave(save_path, pc_recon, check_contrast=False)

    save_path = io_int.save_tiff_file_name(file_name + '_reconstruct_smooth.tif')
    skio.imsave(save_path, smooth_recon, check_contrast=False)

    save_path = io_int.save_tiff_file_name(file_name + '_reconstruct_level.tif')
    skio.imsave(save_path, level_recon, check_contrast=False)


if __name__ == '__main__':
    main()
