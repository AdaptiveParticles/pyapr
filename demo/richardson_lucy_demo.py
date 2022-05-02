import pyapr
from time import time

"""
Read a selected APR from file and apply Richardson-Lucy deconvolution
"""

# Get input APR file path from gui
io_int = pyapr.utils.InteractiveIO()
fpath_apr = io_int.get_apr_file_name()

# Read from APR file
apr, parts = pyapr.io.read(fpath_apr)

# Copy particles to float
parts = pyapr.FloatParticles(parts)

# Add a small offset to the particle values to avoid division by 0
offset = 1e-5 * parts.max()
parts += offset

# Specify the PSF and number of iterations
psf = pyapr.filter.get_gaussian_stencil(size=5, sigma=1, ndims=3, normalize=True)
niter = 10

# Richardson-lucy deconvolution
t0 = time()
output = pyapr.FloatParticles()
pyapr.restoration.richardson_lucy(apr, parts, output, psf, niter, use_stencil_downsample=True,
                                  normalize_stencil=True, resume=False)
print('RL took {} seconds'.format(time()-t0))


# Using total variation regularization
t0 = time()
output_tv = pyapr.FloatParticles()
reg_factor = 1e-2
pyapr.restoration.richardson_lucy_tv(apr, parts, output_tv, psf, niter, reg_factor, use_stencil_downsample=True,
                                     normalize_stencil=True, resume=False)
print('RLTV took {} seconds'.format(time()-t0))


# Alternatively, if PyLibAPR is built with CUDA enabled and psf is of size (3, 3, 3) or (5, 5, 5)
cuda = False
if pyapr.cuda_enabled() and psf.shape in [(3, 3, 3), (5, 5, 5)]:
    t0 = time()
    output_cuda = pyapr.FloatParticles()
    pyapr.restoration.richardson_lucy_cuda(apr, parts, output_cuda, psf, niter, use_stencil_downsample=True,
                                           normalize_stencil=True, resume=False)
    print('RL cuda took {} seconds'.format(time()-t0))
    cuda = True


# Display the result
pyapr.viewer.parts_viewer(apr, output)
pyapr.viewer.parts_viewer(apr, output_tv)
if cuda:
    pyapr.viewer.parts_viewer(apr, output_cuda)
