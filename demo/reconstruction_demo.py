import pyapr
import numpy as np
from skimage import io as skio


def main():

    io_int = pyapr.filegui.InteractiveIO()
    fpath_apr = io_int.get_apr_file_name()  # get APR file path from gui

    aprfile = pyapr.io.APRFile()
    aprfile.set_read_write_tree(True)

    # Initialize APR and particle objects
    parts = pyapr.ShortParticles()
    apr = pyapr.APR()

    # Read from APR file
    aprfile.open(fpath_apr, 'READ')
    aprfile.read_apr(apr)
    aprfile.read_particles(apr, 'particles', parts)
    aprfile.close()

    recon = np.array(pyapr.numerics.reconstruction.recon_pc(apr, parts), copy=False)
    smooth_recon = np.array(pyapr.numerics.reconstruction.recon_smooth(apr, parts), copy=False)
    level_recon = np.array(pyapr.numerics.reconstruction.recon_level(apr), copy=False)

    # save the results
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
