import pyapr
from time import time
"""
This demo reads an APR, applies a convolution operation and displays the result
"""

io_int = pyapr.filegui.InteractiveIO()
fpath_apr = io_int.get_apr_file_name()  # get APR file path from gui

# Read from APR file
apr, parts = pyapr.io.read(fpath_apr)

# Stencil and output must be float32
stencil = pyapr.numerics.filter.get_gaussian_stencil(size=5, sigma=1, ndims=3, normalize=True)
out = pyapr.FloatParticles()

# Convolve using CPU:
t0 = time()
pyapr.numerics.convolve(apr, parts, out, stencil, use_stencil_downsample=True,
                        normalize_stencil=True, use_reflective_boundary=False)
print('convolve took {} seconds'.format(time()-t0))


# Alternative CPU convolution algorithm:
t0 = time()
pyapr.numerics.convolve_pencil(apr, parts, out, stencil, use_stencil_downsample=True,
                               normalize_stencil=True, use_reflective_boundary=False)
print('convolve_pencil took {} seconds'.format(time()-t0))


# Convolve using GPU (stencil must be of shape 3x3x3 or 5x5x5):
if pyapr.cuda_build() and stencil.shape in [(3, 3, 3), (5, 5, 5)]:
    t0 = time()
    pyapr.numerics.convolve_cuda(apr, parts, out, stencil, use_stencil_downsample=True,
                                 normalize_stencil=True, use_reflective_boundary=False)
    print('convolve_cuda took {} seconds'.format(time()-t0))

# Display the result
pyapr.viewer.parts_viewer(apr, out)
