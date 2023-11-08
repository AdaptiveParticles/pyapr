import os
import pytest
import pyapr
from .helpers import load_test_apr, get_test_apr_path
import numpy as np


RECON_MODES = [
    'constant',
    'smooth',
    'level'
]


@pytest.mark.parametrize("mode", RECON_MODES)
@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_reconstruction(mode, ndim):
    apr, parts = load_test_apr(ndim)
    fpath = get_test_apr_path(ndim)

    slicer = pyapr.reconstruction.APRSlicer(apr, parts, mode=mode)
    lazy_slicer = pyapr.reconstruction.LazySlicer(fpath, mode=mode)

    assert slicer.ndim == lazy_slicer.ndim == 3
    assert slicer.shape == lazy_slicer.shape == apr.shape()

    for level_delta in (-2, -1, 0, 1):
        slicer.set_level_delta(level_delta)
        lazy_slicer.set_level_delta(level_delta)

        rc1 = slicer[:]
        rc2 = lazy_slicer[:]
        assert rc1.shape == rc2.shape
        assert np.allclose(rc1, rc2)

        patch = pyapr.ReconPatch()
        patch.level_delta = level_delta
        rc2 = pyapr.reconstruction.reconstruct_lazy(fpath, patch=patch, mode=mode).squeeze()
        assert rc1.shape == rc2.shape
        assert np.allclose(rc1, rc2)

        assert np.allclose(slicer[0], lazy_slicer[0])
        assert np.allclose(slicer[0, float(0), :], lazy_slicer[0, float(0), :])

    assert np.max(slicer) == parts.max() == np.max(np.array(parts))

    slicer = slicer.astype(np.float32)
    assert isinstance(slicer.parts, pyapr.FloatParticles)

    slicer = slicer.astype(int)
    assert isinstance(slicer.parts, pyapr.IntParticles)

    with pytest.raises(ValueError):
        slicer = slicer.astype(np.int8)