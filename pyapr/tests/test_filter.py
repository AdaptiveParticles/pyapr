import pytest
import pyapr
from .helpers import load_test_apr
import numpy as np


@pytest.mark.filterwarnings("ignore:Method \'cuda\'")
@pytest.mark.parametrize("parts_type", [pyapr.ShortParticles, pyapr.FloatParticles])
@pytest.mark.parametrize("stencil_shape", [(3, 3, 5), (1, 5, 7), (1, 1, 13)])
def test_convolution(parts_type, stencil_shape):
    apr, parts = load_test_apr(3)
    parts = parts_type(parts)

    stencil = np.arange(1, np.prod(stencil_shape)+1).reshape(stencil_shape)

    for op in (pyapr.filter.convolve, pyapr.filter.correlate):
        res1 = op(apr, parts, stencil, method='pencil')

        res2 = pyapr.FloatParticles(apr.total_number_particles())
        res2 = op(apr, parts, stencil, method='slice', output=res2)
        assert res1 == res2

        res2 = op(apr, parts, stencil, method='cuda')
        assert res2 == res1


@pytest.mark.parametrize("parts_type", [pyapr.ByteParticles, pyapr.ShortParticles, pyapr.FloatParticles])
def test_gradient_filters(parts_type):
    apr, parts = load_test_apr(3)
    parts = parts_type(parts)

    dy = pyapr.filter.gradient(apr, parts, dim=0, delta=0.7)
    dx = pyapr.filter.gradient(apr, parts, dim=1, delta=0.9)
    dz = pyapr.filter.gradient(apr, parts, dim=2, delta=1.1)
    gradmag_manual = np.sqrt(np.array(dz*dz + dx*dx + dy*dy))

    gradmag = pyapr.filter.gradient_magnitude(apr, parts, deltas=(0.7, 0.9, 1.1))
    assert np.allclose(np.array(gradmag), gradmag_manual)

    dy = pyapr.filter.sobel(apr, parts, dim=0, delta=1.3)
    dx = pyapr.filter.sobel(apr, parts, dim=1, delta=1.5)
    dz = pyapr.filter.sobel(apr, parts, dim=2, delta=0.9)
    gradmag_manual = np.sqrt(np.array(dz*dz + dx*dx + dy*dy))

    gradmag = pyapr.filter.sobel_magnitude(apr, parts, deltas=(1.3, 1.5, 0.9))
    assert np.allclose(np.array(gradmag), gradmag_manual)


@pytest.mark.parametrize("parts_type", [pyapr.ShortParticles, pyapr.FloatParticles])
@pytest.mark.parametrize("filter_size", [(3, 3, 3), (1, 5, 5)])
def test_rank_filters(parts_type, filter_size):
    apr, parts = load_test_apr(3)
    parts = parts_type(parts)

    med_output = pyapr.filter.median_filter(apr, parts, filter_size)
    min_output = pyapr.filter.min_filter(apr, parts, filter_size)
    max_output = pyapr.filter.max_filter(apr, parts, filter_size)

    assert med_output[9123] in parts
    assert min_output.min() == parts.min()
    assert max_output.max() == parts.max()


@pytest.mark.parametrize("parts_type", [pyapr.ByteParticles, pyapr.ShortParticles, pyapr.FloatParticles])
@pytest.mark.parametrize("filter_size", [(3, 3, 5), (1, 7, 9)])
def test_std_filter(parts_type, filter_size):
    apr, parts = load_test_apr(3)
    parts = parts_type(parts)
    output = pyapr.filter.std(apr, parts, filter_size)
