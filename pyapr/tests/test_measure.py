import pytest
import pyapr
from .helpers import load_test_apr_obj
import numpy as np


MASK_TYPES = [
    pyapr.ByteParticles,
    pyapr.ShortParticles,
    pyapr.LongParticles
]


@pytest.mark.parametrize("mask_type", MASK_TYPES)
def test_measures(mask_type):
    # load apr and generate binary mask
    apr, parts = load_test_apr_obj()
    mask = parts > 101

    # find object labels
    cc = pyapr.measure.connected_component(apr, mask, output=mask_type())
    assert cc.max() == 2

    # find bounding boxes around each object/label
    min_coords, max_coords = pyapr.measure.find_objects(apr, cc)
    assert min_coords.shape == max_coords.shape == (3, 3)

    # compute the (weighted) volumetric center of each object
    obj_centers = pyapr.measure.find_label_centers(apr, cc)
    obj_centers_weighted = pyapr.measure.find_label_centers(apr, cc, weights=parts)
    assert obj_centers.shape == obj_centers_weighted.shape == (3, 3)

    # check that object centers are within the bounding boxes computed by `find_objects`
    for i in (1, 2):
        for j in range(3):
            assert min_coords[i, j] < obj_centers[i, j] < max_coords[i, j]
            assert min_coords[i, j] < obj_centers_weighted[i, j] < max_coords[i, j]

    # compute the volume of each object
    vol = pyapr.measure.find_label_volume(apr, cc)

    # compute volumes on reconstructions for comparison
    slicer = pyapr.reconstruction.APRSlicer(apr, cc)
    for obj in (1, 2):
        patch = slicer[min_coords[obj, 0]:max_coords[obj, 0],
                       min_coords[obj, 1]:max_coords[obj, 1],
                       min_coords[obj, 2]:max_coords[obj, 2]]
        assert vol[obj] == np.sum(patch == obj)



