from _pyaprwrapper.data_containers import APR, ByteParticles, ShortParticles, FloatParticles, LongParticles
from typing import Union, Optional, List, Tuple


def _check_input(apr: APR,
                 parts: Union[ByteParticles, ShortParticles, FloatParticles, LongParticles],
                 allowed_types: Optional[Union[List, Tuple]] = None):
    if allowed_types:
        if not isinstance(parts, tuple(allowed_types)):
            raise TypeError(f'Input particles must be of type {allowed_types}, got {type(parts)}.')
    if apr.total_number_particles() == 0:
        raise ValueError(f'Input APR {apr} is not initialized.')
    if len(parts) != apr.total_number_particles():
        raise ValueError(f'Size mismatch between input APR: {apr} and particles: {parts}.')


def _check_input_tree(apr: APR,
                      tree_parts: Union[ByteParticles, ShortParticles, FloatParticles, LongParticles],
                      allowed_types: Optional[Union[List, Tuple]] = None):
    if allowed_types:
        if not isinstance(tree_parts, tuple(allowed_types)):
            raise TypeError(f'Input tree particles must be of type {allowed_types}, got {type(tree_parts)}.')
    if apr.total_number_particles() == 0:
        raise ValueError(f'Input APR {apr} is not initialized.')
    if len(tree_parts) != apr.total_number_tree_particles():
        raise ValueError(f'Size mismatch between input APR ({apr.total_number_tree_particles()} tree particles) and '
                         f'input tree particles: {tree_parts}.')
