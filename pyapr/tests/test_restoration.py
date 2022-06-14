import os
import pytest
import pyapr
from .helpers import load_test_apr
import numpy as np

PARTICLE_TYPES = [
    pyapr.ByteParticles,
    pyapr.ShortParticles,
    pyapr.FloatParticles
]


@pytest.mark.filterwarnings('ignore:richardson_lucy_cuda')
@pytest.mark.parametrize("parts_type", PARTICLE_TYPES)
@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_richardson_lucy(parts_type, ndim):
    apr, parts = load_test_apr(ndim)
    parts = parts_type(parts)

    psf = pyapr.filter.get_gaussian_stencil(size=5, sigma=0.8, ndims=ndim, normalize=True)
    niter = 10

    rl_out = pyapr.restoration.richardson_lucy(apr, parts, psf, num_iter=niter)

    if ndim == 3:
        rl_cuda = pyapr.restoration.richardson_lucy_cuda(apr, parts, psf, num_iter=niter)
        # should give the same result as richardson_lucy on cpu
        assert np.allclose(np.array(rl_out), np.array(rl_cuda))

    rl_out = pyapr.restoration.richardson_lucy_tv(apr, parts, psf, num_iter=niter, resume=True, output=rl_out)

    with pytest.raises(ValueError):
        # resume with wrongly initialized output
        pyapr.restoration.richardson_lucy(apr, parts, psf, num_iter=niter, resume=True, output=pyapr.FloatParticles(10))
