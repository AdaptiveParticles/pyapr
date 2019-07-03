import pyapr
import pyapr.nn as aprnn
import numpy as np
from demo.io import read_tiff
import unittest
import pyapr.nn.testing as testing


class TestAPRNetModules(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestAPRNetModules, self).__init__(*args, **kwargs)

        # Load in an image and extract a small tile
        img = read_tiff('../../LibAPR/test/files/Apr/sphere_120/sphere_original.tif')
        img = img[20:52, 40:72, 53].astype(np.float32)

        # Ensure that pixel values are distinct to avoid problems with finite difference of max pooling
        for i in range(32):
            for j in range(32):
                img[i, j] += 0.001 * (i*32 + j)

        # Initialize objects for APR conversion
        apr = pyapr.APR()
        parts = pyapr.FloatParticles()
        par = pyapr.APRParameters()
        converter = pyapr.converter.FloatConverter()

        # Set some parameters
        par.auto_parameters = False
        par.rel_error = 0.1
        par.Ip_th = 0
        par.gradient_smoothing = 2
        par.sigma_th = 50
        par.sigma_th_max = 20
        converter.set_parameters(par)
        converter.set_verbose(False)

        # Compute APR and sample particle values
        converter.get_apr(apr, img)
        parts.sample_image(apr, img)

        apr_arr = np.empty(1, dtype=object)
        parts_arr = np.empty(1, dtype=object)
        apr_arr[0] = apr
        parts_arr[0] = parts

        x, dlvl = aprnn.APRInputLayer()(apr_arr, parts_arr, dtype=np.float64)
        x.requires_grad = True

        self.aprs = apr_arr
        self.x = x
        self.dlvl = dlvl

    def test_gradients_maxpool(self):
        m = aprnn.APRMaxPool(increment_level_delta=False)
        assert testing.gradcheck(m, (self.x, self.aprs, self.dlvl))

    def test_gradients_conv1x1(self):
        m = aprnn.APRConv(1, 4, 1, 2)
        assert testing.gradcheck(m, (self.x, self.aprs, self.dlvl))

    def test_gradients_conv3x3(self):
        m = aprnn.APRConv(1, 4, 3, 2)
        assert testing.gradcheck(m, (self.x, self.aprs, self.dlvl))

    def test_gradients_conv5x5(self):
        m = aprnn.APRConv(1, 4, 5, 2)
        assert testing.gradcheck(m, (self.x, self.aprs, self.dlvl))


if __name__ == '__main__':
    unittest.main()


