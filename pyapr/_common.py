from _pyaprwrapper.data_containers import APR, ByteParticles, ShortParticles, FloatParticles, LongParticles
from typing import Union, Optional, List, Tuple


def _check_input(apr: APR,
                 parts: Union[ByteParticles, ShortParticles, FloatParticles, LongParticles],
                 allowed_types: Optional[Union[List, Tuple]] = None):
    if allowed_types:
        assert isinstance(parts, tuple(allowed_types)), \
            TypeError(f'Input particles must be of type {allowed_types}, got {type(parts)}.')
    assert apr.total_number_particles() > 0, ValueError(f'Input APR is not initialized.')
    assert len(parts) == apr.total_number_particles(), \
        ValueError(f'Size mismatch between input APR: {apr} and particles: {parts}.')


def _check_input_tree(apr: APR,
                      tree_parts: Union[ByteParticles, ShortParticles, FloatParticles, LongParticles],
                      allowed_types: Optional[Union[List, Tuple]] = None):
    if allowed_types:
        assert isinstance(tree_parts, tuple(allowed_types)), \
            TypeError(f'Input tree particles must be of type {allowed_types}, got {type(tree_parts)}.')
    assert apr.total_number_particles() > 0, ValueError(f'Input APR is not initialized.')
    assert len(tree_parts) == apr.total_number_tree_particles(), \
        ValueError(f'Size mismatch between input APR: ({apr.total_number_tree_particles()} tree particles) and '
                   f'tree_parts: {tree_parts}.')
