import pytest
import pyapr
from .helpers import load_test_apr_obj
import numpy as np

PARTICLE_TYPES = [
    pyapr.ShortParticles,
    pyapr.FloatParticles
]


@pytest.mark.parametrize("parts_type", PARTICLE_TYPES)
@pytest.mark.parametrize("constant_neighbor_scale", [True, False])
def test_graphcut(parts_type, constant_neighbor_scale):
    apr, parts = load_test_apr_obj()
    parts = parts_type(parts)

    # this image is trivially segmented by thresholding
    gt_mask = parts > 100

    # test graphcut
    mask = pyapr.segmentation.graphcut(apr, parts, intensity_threshold=101, beta=3.0, push_depth=1,
                                       constant_neighbor_scale=constant_neighbor_scale)
    assert mask == gt_mask

    # test blocked version
    mask = pyapr.segmentation.graphcut(apr, parts, intensity_threshold=101, beta=3.0, z_block_size=16, z_ghost_size=32,
                                       push_depth=1, constant_neighbor_scale=constant_neighbor_scale)
    assert mask == gt_mask

    # run compute_terminal_costs
    foreground, background = pyapr.segmentation.compute_terminal_costs(apr, parts)

    with pytest.raises(TypeError):
        # unsupported output type
        mask = pyapr.segmentation.graphcut(apr, parts, output=pyapr.FloatParticles())
