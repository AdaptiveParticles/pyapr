from _pyaprwrapper.data_containers import APR, ShortParticles, LongParticles, ByteParticles
from _pyaprwrapper.measure import connected_component as _connected_component
from .._common import _check_input
from typing import Union, Optional

__allowed_types__ = (ByteParticles, ShortParticles, LongParticles)


def connected_component(apr: APR,
                        mask: Union[ByteParticles, ShortParticles, LongParticles],
                        output: Optional[Union[ByteParticles, ShortParticles, LongParticles]] = None) \
        -> Union[ByteParticles, ShortParticles, LongParticles]:
    """
    Label the connected components of an input particle mask. Two particles are considered connected if they
    are face-side neighbors and take non-zero values.

    Parameters
    ----------
    apr: APR
        Input APR data structure.
    mask: ByteParticles, ShortParticles or LongParticles
        Input (binary) particle mask.
    output: ByteParticles, ShortParticles or LongParticles, optional
        (optional) Particle object for the output labels. If not provided, a LongParticles (uint64) object is generated
        and returned. (default: None)

    Returns
    -------
    output: ByteParticles, ShortParticles or LongParticles
        Particle data containing the connected component labels
    """
    _check_input(apr, mask, __allowed_types__)
    if output is None:
        output = LongParticles()

    assert isinstance(output, __allowed_types__), \
        TypeError(f'output (if provided) must be of type {__allowed_types__}, received {type(output)}.')

    _connected_component(apr, mask, output)
    return output
