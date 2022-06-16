from _pyaprwrapper.data_containers import APR, ByteParticles, ShortParticles, LongParticles
import _pyaprwrapper.measure as _measure
from .._common import _check_input
import numpy as np
from typing import Union, Tuple

__allowed_types__ = (ByteParticles, ShortParticles, LongParticles)


def find_objects(apr: APR,
                 labels: Union[ByteParticles, ShortParticles, LongParticles]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find and return tight bounding boxes for each unique input label. Assumes that the labels are
    ordered from 0, such that 0 is background and each value > 0 corresponds to a connected component.

    Parameters
    ----------
    apr: APR
        Input APR data structure.
    labels: ByteParticles, ShortParticles or LongParticles
        Input (object) labels.

    Returns
    -------
    min_coords: numpy.ndarray
        Array of shape `(labels.max() + 1, 3)` containing the "lower" corner of each bounding box in z, x and y.
    max_coords: numpy.ndarray
        Array of shape `(labels.max() + 1, 3)` containing the "upper" corner of each bounding box in z, x and y.
    """
    _check_input(apr, labels, __allowed_types__)
    max_label = labels.max()
    max_dim = max([apr.org_dims(x) for x in range(3)])
    min_coords = np.full((max_label+1, 3), max_dim+1, dtype=np.int32)
    max_coords = np.zeros((max_label+1, 3), dtype=np.int32)
    _measure.find_objects(apr, labels, min_coords, max_coords)

    max_coords[0, :] = apr.shape()
    min_coords[0, :] = 0

    return min_coords, max_coords
