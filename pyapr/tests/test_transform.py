import pytest
import pyapr
from .helpers import load_test_apr
import numpy as np

PARTICLE_TYPES = [
    pyapr.ByteParticles,
    pyapr.ShortParticles,
    pyapr.FloatParticles,
    pyapr.LongParticles
]

PROJECTION_METHODS = [
    'direct',
    'pyramid'
]


@pytest.mark.filterwarnings('ignore:max projection')
@pytest.mark.parametrize("parts_type", PARTICLE_TYPES)
@pytest.mark.parametrize("method", PROJECTION_METHODS)
def test_maximum_projection(parts_type, method):
    apr, parts = load_test_apr(3)
    parts = parts_type(parts)

    for dim in (0, 1, 2):
        res = pyapr.transform.maximum_projection(apr, parts, dim, method=method)
        recon = pyapr.reconstruction.reconstruct_constant(apr, parts)
        assert np.allclose(res, np.max(recon, axis=2-dim))

        patch = pyapr.ReconPatch()
        _shape = apr.shape()
        patch.z_begin = 1
        patch.z_end = _shape[0] // 2
        patch.x_begin = 2
        patch.x_end = _shape[1] // 2
        patch.y_begin = 3
        patch.y_end = _shape[2] // 2

        res = pyapr.transform.maximum_projection(apr, parts, dim, patch=patch, method=method)
        recon = pyapr.reconstruction.reconstruct_constant(apr, parts, patch=patch)
        assert np.allclose(res, np.max(recon, axis=2-dim))

    with pytest.raises(ValueError):
        # invalid dim argument
        res = pyapr.transform.maximum_projection(apr, parts, dim=3, method=method)

    with pytest.raises(ValueError):
        # invalid patch specification (z_begin > z_end)
        patch = pyapr.ReconPatch()
        patch.z_begin = 5; patch.z_end = 3
        patch.level_delta = -1
        res = pyapr.transform.maximum_projection(apr, parts, dim=0, patch=patch, method=method)

