import os
import pyapr
from skimage import io as skio


"""
This demo shows how to convert an image to APR using a fixed set of parameters.
"""

# Read in an image
io_int = pyapr.utils.InteractiveIO()
fpath = io_int.get_tiff_file_name()  # get image file path from gui (data type must be float32 or uint16)
img = skio.imread(fpath)

# Set some parameters
par = pyapr.APRParameters()
par.rel_error = 0.1          # relative error threshold
par.gradient_smoothing = 3   # b-spline smoothing parameter for gradient estimation
#                              0 = no smoothing, higher = more smoothing
par.dx = 1
par.dy = 1                   # voxel size
par.dz = 1
# threshold parameters
par.Ip_th = 0                # regions below this intensity are regarded as background
par.grad_th = 3              # gradients below this value are set to 0
par.sigma_th = 10            # the local intensity scale is clipped from below to this value
par.auto_parameters = True   # if true, 'grad_th' and 'sigma_th' are computed automatically based on histograms

# Compute APR and sample particle values
apr, parts = pyapr.converter.get_apr(img, params=par, verbose=True)

# Display the APR
pyapr.viewer.parts_viewer(apr, parts)

# Write the resulting APR to file
print("Writing APR to file ... \n")
fpath_apr = io_int.save_apr_file_name()  # get path through gui
pyapr.io.write(fpath_apr, apr, parts)

if fpath_apr:
    # Display the size of the file
    file_sz = os.path.getsize(fpath_apr)
    print("APR File Size: {:7.2f} MB \n".format(file_sz * 1e-6))

    # Compute compression ratio
    mcr = os.path.getsize(fpath) / file_sz
    print("Memory Compression Ratio: {:7.2f}".format(mcr))
