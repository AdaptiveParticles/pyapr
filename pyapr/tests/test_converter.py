import pytest
from os.path import join
import pyapr
from .helpers import load_test_image, constant_upsample, expand
import numpy as np
from skimage import io as skio

IMAGES = [load_test_image(d) for d in (1, 2, 3)]
IMAGES[1] = IMAGES[1].astype(np.float32)
ERR_THRESHOLDS = [0.2, 0.1, 0.05, 0.01]
UNSUPPORTED_TYPES = [np.int16, np.int32, np.float64]


@pytest.mark.parametrize("img", IMAGES)
@pytest.mark.parametrize("rel_error", ERR_THRESHOLDS)
def test_get_apr(tmpdir, rel_error: float, img: np.ndarray):

    # set conversion parameters
    par = pyapr.APRParameters()
    par.rel_error = rel_error
    par.auto_parameters = True
    par.output_steps = True
    par.output_dir = str(tmpdir) + '/'

    # convert image to APR
    apr, parts = pyapr.converter.get_apr(img, params=par, verbose=True)

    # load pipeline steps
    lis = expand(skio.imread(join(par.output_dir, 'local_intensity_scale_rescaled.tif'))).astype(np.float32)
    grad = expand(skio.imread(join(par.output_dir, 'gradient_step.tif'))).astype(np.float32)

    # upsample lis and gradient
    lis = constant_upsample(lis, apr.shape())
    grad = constant_upsample(grad, apr.shape())

    # check constant reconstruction error
    recon = pyapr.reconstruction.reconstruct_constant(apr, parts).astype(np.float32)
    img = img.astype(np.float32)
    err = np.divide(np.abs(img - recon), lis)

    # ignore regions below pipeline threshold parameters
    par = apr.get_parameters()
    err[grad < par.grad_th] = 0
    err[img < par.Ip_th] = 0

    # check maximum error
    max_error = err.max()
    assert max_error < rel_error


@pytest.mark.parametrize("dtype", UNSUPPORTED_TYPES)
def test_get_apr_type_error(dtype):
    img = IMAGES[1].astype(dtype)
    with pytest.raises(TypeError):
        apr, parts = pyapr.converter.get_apr(img)

