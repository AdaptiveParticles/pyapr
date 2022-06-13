from _pyaprwrapper.tree import sample_from_tree as _sample_from_tree
from _pyaprwrapper.data_containers import APR, ByteParticles, ShortParticles, FloatParticles, LongParticles
from .._common import _check_input, _check_input_tree
from typing import Union

ParticleData = Union[ByteParticles, ShortParticles, FloatParticles, LongParticles]


def sample_from_tree(apr: APR,
                     parts: ParticleData,
                     tree_parts: ParticleData,
                     num_levels: int = 1,
                     in_place: bool = False):
    """
    Coarsen particle values by sampling from parent nodes in the APR tree. Optionally further coarsen the finest
    particle values by setting `num_levels>1`.

    Parameters
    ----------
    apr: APR
        Input APR data structure.
    parts: ByteParticles, ShortParticles, FloatParticles or LongParticles
        Input APR particle values.
    tree_parts: ByteParticles, ShortParticles, FloatParticles or LongParticles
        Input APR tree values. Must either be of type FloatParticles or the same type as ``parts``.
    num_levels: int
        Sample values from level ``apr.level_max()-num_levels``. If ``num_levels=1``, each particle takes the value
        of its parent node in the APR tree. If ``num_levels>1``, the ``num_levels-1`` finest tree levels are coarsened
        prior to re-sampling particle values. Thus, ``num_levels`` sets the maximum resolution of the sampling. (Default: 1)
    in_place: bool
        If True, both ``parts`` and ``tree_parts`` are modified in-place. (Default: False)

    Returns
    -------
    output: ByteParticles, ShortParticles, FloatParticles or LongParticles
        The resampled particle values.
    """
    _check_input(apr, parts)
    _check_input_tree(apr, tree_parts, (FloatParticles, type(parts)))
    parts = parts if in_place else parts.copy()

    if num_levels <= 0:
        return parts

    tree_parts = tree_parts if in_place else tree_parts.copy()
    _sample_from_tree(apr, parts, tree_parts, num_levels-1)
    return parts
