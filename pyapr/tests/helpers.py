import os
import pyapr
from skimage import io as skio
import numpy as np


def constant_upsample(img, output_shape, factor=2):
    output = np.zeros(shape=output_shape, dtype=img.dtype)
    for i in range(output_shape[0]):
        for j in range(output_shape[1]):
            for k in range(output_shape[2]):
                output[i, j, k] = img[i//factor, j//factor, k//factor]
    return output


def expand(img):
    while img.ndim < 3:
        img = np.expand_dims(img, axis=0)
    return img


def load_test_apr(dims: int = 3):
    """
    read APR data for testing

    Parameters
    ----------
    dims: int
        dimensionality of the image (1-3)
    """
    assert dims == 1 or dims == 2 or dims == 3, ValueError('\'dims\' must be 1, 2 or 3')
    fpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_files', f'sphere_{dims}D.apr')
    return pyapr.io.read(fpath)


def get_test_apr_path(dims: int = 3):
    """
    return the path of an APR file

    Parameters
    ----------
    dims: int
        dimensionality of the image (1-3)
    """
    assert dims == 1 or dims == 2 or dims == 3, ValueError('\'dims\' must be 1, 2 or 3')
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_files', f'sphere_{dims}D.apr')


def load_test_image(dims: int = 3):
    """
    read pixel image for testing

    Parameters
    ----------
    dims: int
        dimensionality of the image (1-3)
    """
    assert dims == 1 or dims == 2 or dims == 3, ValueError('\'dims\' must be 1, 2 or 3')
    fpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_files', f'sphere_{dims}D.tif')
    return expand(skio.imread(fpath))


def load_test_apr_obj():
    """
    return test APR with two objects
    """
    fpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_files', 'two_objects.apr')
    return pyapr.io.read(fpath)
