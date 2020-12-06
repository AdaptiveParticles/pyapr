import unittest
import pyapr
from skimage import io as skio
import numpy as np
import os


class BasicTests(unittest.TestCase):

    def setUp(self):
        self.this_dir = os.path.dirname(os.path.abspath(__file__))
        self.impath_3D = os.path.abspath(os.path.join(self.this_dir, '../../LibAPR/test/files/Apr/sphere_120/sphere_original.tif'))
        self.impath_2D = os.path.abspath(os.path.join(self.this_dir, '../../LibAPR/test/files/Apr/sphere_2D/original.tif'))
        self.impath_1D = os.path.abspath(os.path.join(self.this_dir, '../../LibAPR/test/files/Apr/sphere_1D/original.tif'))

        self.apr_parameters = pyapr.APRParameters()
        self.apr_parameters.sigma_th = 30
        self.apr_parameters.grad_th = 15

    def __convert_image(self, impath, dtype):
        img = skio.imread(impath).astype(dtype)
        apr, parts = pyapr.converter.get_apr(img, verbose=False, params=self.apr_parameters)

        # TODO: use output_steps flag and check reconstruction condition

        self.assertGreater(apr.computational_ratio(), 1)
        self.assertGreater(len(parts), 1)

    def test_get_apr(self):
        self.__convert_image(self.impath_1D, np.uint16)
        self.__convert_image(self.impath_2D, np.uint16)
        self.__convert_image(self.impath_3D, np.uint16)

        self.__convert_image(self.impath_1D, np.float32)
        self.__convert_image(self.impath_2D, np.float32)
        self.__convert_image(self.impath_3D, np.float32)

        with self.assertRaises(TypeError):
            self.__convert_image(self.impath_1D, np.float64)

        with self.assertRaises(TypeError):
            self.__convert_image(self.impath_1D, np.uint8)

        with self.assertRaises(TypeError):
            self.__convert_image(self.impath_1D, int)

    def test_io(self):

        # writes files to this path, later removes the file # TODO: find a better way to do this
        fpath = os.path.join(self.this_dir, 'temporary_apr_file.apr')

        for impath in (self.impath_1D, self.impath_2D, self.impath_3D):
            img = skio.imread(impath)
            apr, parts = pyapr.converter.get_apr(img, verbose=False, params=self.apr_parameters)

            pyapr.io.write(fpath, apr, parts)

            apr2 = pyapr.APR()
            parts2 = pyapr.ShortParticles()

            pyapr.io.read(fpath, apr2, parts2)

            self.assertEqual(apr.org_dims(), apr2.org_dims())
            self.assertEqual(apr.total_number_particles(), apr2.total_number_particles())
            self.assertEqual(apr.total_number_particles(), len(parts2))

            self.assertTrue(
                np.alltrue(
                    np.array(parts, copy=False) == np.array(parts2, copy=False)
                )
            )

        os.remove(fpath)    # remove temporary file


if __name__ == '__main__':
    unittest.main()
