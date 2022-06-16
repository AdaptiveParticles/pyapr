import pytest
import pyapr
from .helpers import load_test_apr, load_test_apr_obj
import numpy as np

PARTICLE_TYPES = [
    pyapr.ByteParticles,
    pyapr.ShortParticles,
    pyapr.FloatParticles,
    pyapr.LongParticles
]

MASK_TYPES = [
    pyapr.ByteParticles,
    pyapr.ShortParticles,
    pyapr.LongParticles
]


@pytest.mark.parametrize("parts_type", PARTICLE_TYPES)
def test_erosion_dilation(parts_type):
    apr, parts = load_test_apr_obj()
    mask = parts_type(apr.total_number_particles())
    mask.fill(0)

    indices = np.random.randint(apr.total_number_particles(), size=20)
    it = apr.iterator()

    for idx in indices:
        # set a single particle to 1 and dilate
        mask[idx] = 1
        tmp = pyapr.morphology.dilation(apr, mask, binary=True, inplace=False)

        # get coordinates to find neighbors
        level, z_l, x_l, y_l = it.find_coordinates(idx)
        level_size = 2**(it.level_max()-level)
        z = z_l * level_size
        x = x_l * level_size
        y = y_l * level_size

        # ensure neighboring values are 1
        if z > 0:
            assert tmp[it.find_particle(z - 1, x, y)] == 1
        if x > 0:
            assert tmp[it.find_particle(z, x - 1, y)] == 1
        if y > 0:
            assert tmp[it.find_particle(z, x, y - 1)] == 1
        if z + level_size < apr.org_dims(2):
            assert tmp[it.find_particle(z + level_size, x, y)] == 1
        if x + level_size < apr.org_dims(1):
            assert tmp[it.find_particle(z, x + level_size, y)] == 1
        if y + level_size < apr.org_dims(0):
            assert tmp[it.find_particle(z, x, y + level_size)] == 1

        # after eroding the result, only the original particle should be 1
        tmp = pyapr.morphology.erosion(apr, tmp, binary=True, inplace=True)
        assert tmp[idx] == 1
        assert np.sum(np.array(tmp)) == 1
        mask[idx] = 0


@pytest.mark.parametrize("parts_type", PARTICLE_TYPES)
@pytest.mark.parametrize("binary", [True, False])
def test_run_morphology_ops(parts_type, binary):
    apr, parts = load_test_apr_obj()
    parts = parts_type(parts > 101) if binary else parts_type(parts)

    out = pyapr.morphology.opening(apr, parts, binary=binary, radius=2)
    out = pyapr.morphology.closing(apr, parts, binary=binary, radius=2)
    out = pyapr.morphology.tophat(apr, parts, binary=binary, radius=2)
    out = pyapr.morphology.bottomhat(apr, parts, binary=binary, radius=2)
    out = pyapr.morphology.find_perimeter(apr, parts)


@pytest.mark.parametrize("mask_type", MASK_TYPES)
def test_remove_x_ops(mask_type):
    apr, parts = load_test_apr_obj()
    mask = parts > 101
    cc = pyapr.measure.connected_component(apr, mask, output=mask_type())
    num_obj = cc.max()
    labels = set(range(num_obj+1))

    vol = pyapr.measure.find_label_volume(apr, cc)
    threshold = np.min(vol[1:]) + 1
    min_label = np.argmin(vol[1:]) + 1

    out = pyapr.morphology.remove_small_objects(apr, cc, min_volume=threshold)
    assert min_label not in out
    labels.remove(min_label)
    max_label = max(labels)
    assert max_label in out

    out = pyapr.morphology.remove_large_objects(apr, cc, max_volume=threshold)
    assert min_label in out
    assert max_label not in out

    out = pyapr.morphology.remove_small_holes(apr, cc, 50)
    assert out == cc    # there are no holes in the input mask

    out = pyapr.morphology.remove_edge_objects(apr, cc)
    assert out == cc    # there are no objects on edges in the input mask




