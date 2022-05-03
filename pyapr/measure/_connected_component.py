from _pyaprwrapper.data_containers import APR, ShortParticles, LongParticles, ByteParticles
import _pyaprwrapper.measure as _measure
from typing import Union, Optional


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

    if not isinstance(output, (type(None), ByteParticles, ShortParticles, LongParticles)):
        raise TypeError(f'Invalid argument \'output\' of type {type(output)}. Allowed types are \'NoneType\', '
                        f'ByteParticles, ShortParticles and LongParticles.')

    output = output or LongParticles()
    _measure.connected_component(apr, mask, output)
    return output
