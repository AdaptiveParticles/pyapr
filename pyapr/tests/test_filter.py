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


@pytest.mark.filterwarnings("ignore:Method \'cuda\'")
@pytest.mark.parametrize("parts_type", PARTICLE_TYPES)
@pytest.mark.parametrize("stencil_shape", [(5, 5, 5), (1, 5, 7), (1, 1, 13)])
def test_convolution(parts_type, stencil_shape):
    apr, parts = load_test_apr(3)
    parts = parts_type(parts)

    stencil = np.arange(1, np.prod(stencil_shape)+1).reshape(stencil_shape)
    stencil = stencil / np.sum(stencil)

    for op in (pyapr.filter.convolve, pyapr.filter.correlate):
        res1 = op(apr, parts, stencil, method='pencil')

        res2 = pyapr.FloatParticles(apr.total_number_particles())
        res2 = op(apr, parts, stencil, method='slice', output=res2)
        assert np.allclose(np.array(res1, copy=False), np.array(res2, copy=False))

        res2 = op(apr, parts, stencil, method='cuda')
        assert np.allclose(np.array(res1, copy=False), np.array(res2, copy=False))

        with pytest.raises(ValueError):
            # unsupported method
            res = op(apr, parts, stencil, method='does-not-exist')

        with pytest.raises(TypeError):
            # unsupported parts type
            res = op(apr, (1, 2, 3), stencil)


@pytest.mark.parametrize("parts_type", PARTICLE_TYPES)
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

    with pytest.raises(ValueError):
        # invalid dim argument
        pyapr.filter.gradient(apr, parts, dim=3)

    with pytest.raises(ValueError):
        # invalid dim argument
        pyapr.filter.sobel(apr, parts, dim=3)

    with pytest.raises(ValueError):
        # invalid deltas argument (must be length 3)
        pyapr.filter.gradient_magnitude(apr, parts, deltas=(1, 1))

    with pytest.raises(ValueError):
        # invalid deltas argument (must be length 3)
        pyapr.filter.sobel_magnitude(apr, parts, deltas=(2, ))


@pytest.mark.parametrize("parts_type", PARTICLE_TYPES)
@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_gradient_manual(parts_type, ndim):
    apr, parts = load_test_apr(ndim)
    parts = parts_type(parts)

    # compute y gradient
    grad = pyapr.filter.gradient(apr, parts, dim=0)

    # compute gradient using correlate
    stencil = np.array([-1, 0, 1]).reshape(1, 1, 3) / 2
    grad_manual = pyapr.filter.correlate(apr, parts, stencil, rescale_stencil=True)

    assert np.allclose(np.array(grad), np.array(grad_manual))


@pytest.mark.parametrize("parts_type", PARTICLE_TYPES)
def test_sobel_manual(parts_type):
    apr, parts = load_test_apr(3)
    parts = parts_type(parts)

    # compute sobel gradient
    grad = pyapr.filter.sobel(apr, parts, dim=0)

    # compute gradient using correlate
    stencil = np.outer(np.outer([1, 2, 1], [1, 2, 1]), [-1, 0, 1]).reshape(3, 3, 3) / 32
    methods = ['slice', 'pencil', 'cuda'] if pyapr.cuda_enabled() else ['slice', 'pencil']
    for method in methods:
        grad_manual = pyapr.filter.correlate(apr, parts, stencil, rescale_stencil=True, method=method)
        assert np.allclose(np.array(grad), np.array(grad_manual))


@pytest.mark.parametrize("parts_type", PARTICLE_TYPES)
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

    with pytest.raises(ValueError):
        # unsupported filter size
        res = pyapr.filter.median_filter(apr, parts, (1, 97, 13))


@pytest.mark.parametrize("parts_type", PARTICLE_TYPES)
@pytest.mark.parametrize("filter_size", [3, (1, 7, 9), [3, 3, 5]])
def test_std_filter(parts_type, filter_size):
    apr, parts = load_test_apr(3)
    parts = parts_type(parts)
    output = pyapr.filter.std(apr, parts, filter_size)

    with pytest.raises(ValueError):
        # invalid filter specification (must be int or length 3)
        pyapr.filter.std(apr, parts, (1, 5))
