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


def _check_mean(apr, tree_parts, recon):
    """
    compare tree particles to np.mean() at a single location
    """
    it = apr.tree_iterator()
    idx = apr.total_number_tree_particles() - 10
    level, z_l, x_l, y_l = it.find_coordinates(idx)
    level_size = 2 ** (apr.level_max() - level)
    z = z_l * level_size
    x = x_l * level_size
    y = y_l * level_size
    assert tree_parts[idx] == np.mean(recon[z:z+level_size, x:x+level_size, y:y+level_size])



@pytest.mark.parametrize("parts_type", PARTICLE_TYPES)
@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_fill_tree(parts_type, ndim):
    apr, parts = load_test_apr(ndim)
    parts = parts_type(parts)
    recon = pyapr.reconstruction.reconstruct_constant(apr, parts)

    tree_parts = pyapr.tree.fill_tree_mean(apr, parts, output=pyapr.FloatParticles())
    _check_mean(apr, tree_parts, recon)

    tree_parts = pyapr.tree.fill_tree_max(apr, parts, output=parts_type())
    assert tree_parts[0] == parts.max()

    tree_parts = pyapr.tree.fill_tree_min(apr, parts, output=parts_type())
    assert tree_parts[0] == parts.min()

    for op in (pyapr.tree.fill_tree_mean, pyapr.tree.fill_tree_min, pyapr.tree.fill_tree_max):
        # uninitialized input
        with pytest.raises(ValueError):
            tree_parts = op(pyapr.APR(), pyapr.ShortParticles())

        # unsupported output type
        with pytest.raises(TypeError):
            out = pyapr.ByteParticles() if not isinstance(parts, pyapr.ByteParticles) else pyapr.ShortParticles()
            tree_parts = op(apr, parts, output=out)


@pytest.mark.parametrize("parts_type", PARTICLE_TYPES)
def test_sample_from_tree(parts_type):
    apr, parts = load_test_apr(3)
    parts = parts_type(parts)
    tree_parts = pyapr.tree.fill_tree_mean(apr, parts)

    res = pyapr.tree.sample_from_tree(apr, parts, tree_parts, num_levels=0)
    assert res == parts

    res = pyapr.tree.sample_from_tree(apr, parts, tree_parts, num_levels=2)

    with pytest.raises(TypeError):
        # unsupported tree particles type
        wrong_type = pyapr.ShortParticles if not isinstance(parts, pyapr.ShortParticles) else pyapr.LongParticles
        res = pyapr.tree.sample_from_tree(apr, parts, wrong_type(tree_parts))

    with pytest.raises(ValueError):
        # tree_parts size mismatch
        res = pyapr.tree.sample_from_tree(apr, parts, parts)