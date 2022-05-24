from _pyaprwrapper.data_containers import APR, ShortParticles, LongParticles, FloatParticles, ByteParticles
import _pyaprwrapper.measure as _measure
from .._common import _check_input
import numpy as np
from typing import Union, Optional

__allowed_input_types__ = (ByteParticles, ShortParticles, LongParticles)
__allowed_weight_types__ = (ShortParticles, FloatParticles)


def find_label_centers(apr: APR,
                       labels: Union[ByteParticles, ShortParticles, LongParticles],
                       weights: Optional[Union[ShortParticles, FloatParticles]] = None) -> np.ndarray:
    """
    Compute the volumetric center of each unique input label, optionally weighted by, e.g., image intensity.

    Parameters
    ----------
    apr: APR
        Input APR data structure.
    labels: ByteParticles, ShortParticles or LongParticles
        Input (object) labels.
    weights: ShortParticles or FloatParticles, optional
        (optional) Weight for each particle. Normalization is applied internally. (default: None)

    Returns
    -------
    coords: numpy.ndarray
        Array containing the center coordinates.
    """
    _check_input(apr, labels, __allowed_input_types__)
    max_label = labels.max()
    coords = np.zeros((max_label+1, 3), dtype=np.float64)
    if weights is not None:
        _check_input(apr, weights, __allowed_weight_types__)
        _measure.find_label_centers_weighted(apr, labels, coords, weights)
    else:
        _measure.find_label_centers(apr, labels, coords)
    return coords[np.any(coords > 0, axis=1), :]
