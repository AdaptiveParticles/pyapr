import pyapr
from skimage import io as skio
from pyapr.numerics.reconstruction import reconstruct_constant, reconstruct_smooth, reconstruct_level


"""
This demo illustrates three different pixel image reconstruction methods:

    constant      each pixel takes the value of the particle whose cell contains the pixel
    smooth        additionally smooths regions of coarser resolution to reduce 'blockiness'
    level         each pixel takes the value of the resolution level of the particle cell it belongs to
"""

# get input APR file path from gui
io_int = pyapr.filegui.InteractiveIO()
fpath_apr = io_int.get_apr_file_name()

# Read APR and particles from file
apr, parts = pyapr.io.read(fpath_apr)

pc_recon = reconstruct_constant(apr, parts)     # piecewise constant reconstruction
smooth_recon = reconstruct_smooth(apr, parts)   # smooth reconstruction
level_recon = reconstruct_level(apr)            # level reconstruction

# Save the results
file_name = '.'.join(fpath_apr.split('/')[-1].split('.')[:-1])

save_path = io_int.save_tiff_file_name(file_name + '_reconstruct_const.tif')
if save_path:
    skio.imsave(save_path, pc_recon, check_contrast=False)

save_path = io_int.save_tiff_file_name(file_name + '_reconstruct_smooth.tif')
if save_path:
    skio.imsave(save_path, smooth_recon, check_contrast=False)

save_path = io_int.save_tiff_file_name(file_name + '_reconstruct_level.tif')
if save_path:
    skio.imsave(save_path, level_recon, check_contrast=False)
