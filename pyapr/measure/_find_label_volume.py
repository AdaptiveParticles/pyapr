from _pyaprwrapper.data_containers import APR, ShortParticles, LongParticles, ByteParticles
import _pyaprwrapper.measure as _measure
import numpy as np
from typing import Union


def find_label_volume(apr: APR,
                      labels: Union[ByteParticles, ShortParticles, LongParticles]) -> np.ndarray:
    """
    Return the volume (in voxels) of each unique input label.

    Parameters
    ----------
    apr: APR
        Input APR data structure.
    labels: ByteParticles, ShortParticles or LongParticles
        Input (object) labels.

    Returns
    -------
    volume: numpy.ndarray
        Array of shape `(labels.max() + 1,)` containing the label volumes.
    """

    max_label = labels.max()
    volume = np.zeros((max_label+1), dtype=np.uint64)
    _measure.find_label_volume(apr, labels, volume)
    return volume
