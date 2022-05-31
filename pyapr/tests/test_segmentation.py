import pytest
import pyapr
from .helpers import load_test_apr_obj
import numpy as np

PARTICLE_TYPES = [
    pyapr.ShortParticles,
    pyapr.FloatParticles
]


@pytest.mark.parametrize("parts_type", PARTICLE_TYPES)
def test_graphcut(parts_type):
    apr, parts = load_test_apr_obj()
    parts = parts_type(parts)

    mask = pyapr.segmentation.graphcut(apr, parts, intensity_threshold=101)
    cc = pyapr.measure.connected_component(apr, mask, output=pyapr.ByteParticles())
    assert cc.max() == 2

    # blocked version should give the same result as it takes the entire image into account for each tile
    mask = pyapr.segmentation.graphcut(apr, parts, intensity_threshold=101, z_block_size=16, z_ghost_size=16)
    cc2 = pyapr.measure.connected_component(apr, mask, output=pyapr.ByteParticles())
    if not np.all(np.array(cc2) == np.array(cc)):
        a = np.array(cc2)
        b = np.array(cc)
        diff = a != b
        with np.printoptions(threshold=200):
            print(np.sum(diff))
            print(a[diff])
            print(b[diff])
        assert 0

    foreground, background = pyapr.segmentation.compute_terminal_costs(apr, parts)

    with pytest.raises(TypeError):
        # unsupported output type
        mask = pyapr.segmentation.graphcut(apr, parts, output=pyapr.FloatParticles())
