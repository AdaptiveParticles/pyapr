import pyapr
from time import time

"""
This demo reads an APR, applies a convolution operation and displays the result
"""

io_int = pyapr.utils.InteractiveIO()
fpath_apr = io_int.get_apr_file_name()  # get APR file path from gui

# Read from APR file
apr, parts = pyapr.io.read(fpath_apr)

# Stencil and output must be float32
stencil = pyapr.filter.get_gaussian_stencil(size=5, sigma=1, ndims=3, normalize=True)
out = pyapr.FloatParticles()

# Convolve using CPU:
t0 = time()
out = pyapr.filter.convolve(apr, parts, stencil, output=out, method='slice')
print('convolve (method \'slice\') took {} seconds'.format(time()-t0))


# Alternative CPU convolution algorithm:
t0 = time()
out = pyapr.filter.convolve(apr, parts, stencil, output=out, method='pencil')
print('convolve (method \'pencil\') took {} seconds'.format(time()-t0))


# Convolve using GPU (stencil must be of shape 3x3x3 or 5x5x5):
if pyapr.cuda_enabled() and stencil.shape in [(3, 3, 3), (5, 5, 5)]:
    t0 = time()
    out = pyapr.filter.convolve(apr, parts, stencil, output=out, method='cuda')
    print('convolve (method \'cuda\') took {} seconds'.format(time()-t0))

# Display the result
pyapr.viewer.parts_viewer(apr, out)
