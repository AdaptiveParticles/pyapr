import _pyaprwrapper
import numpy as np


# Prototype APRParameters class
# TODO: decide which parameters should be made visible, improve naming(?) and add descriptions
class APRParameters:
    def __init__(self):
        self.pars = _pyaprwrapper.APRParameters()

    def error_bound(self, val):

        self.pars.rel_error = val

    def intensity_threshold(self, val):

        self.pars.Ip_th = val

    def min_signal(self, val):

        self.pars.min_signal = val

    def sigma_threshold(self, val):

        self.pars.sigma_th = val

    def sigma_threshold_max(self, val):

        self.pars.sigma_th_max = val

    def smoothing(self, val):

        self.pars.lmbda = val

    def auto_parameters(self, val):

        self.pars.auto_parameters = val


# Prototype APR class
# TODO: once particle values are moved out of the APR class, there will be no need for different data types
class APR:
    def __init__(self, dtype=np.float32):
        self.dtype=dtype
        
        if dtype in (np.float32, np.single):
            self.apr = _pyaprwrapper.AprFloat()

        elif dtype == np.uint16:
            self.apr = _pyaprwrapper.AprShort()

        elif dtype == np.uint8:
            self.apr = _pyaprwrapper.AprByte()

        else:
            raise TypeError('APR dtype must be numpy.float32, numpy.uint16 or numpy.uint8')

        self.par = APRParameters() # TODO: should this be here?

    def set_parameters(self, pars):

        if not isinstance(pars, APRParameters):
            raise TypeError('params must be of type APRParameters')

        self.par = pars
        self.apr.set_parameters(self.par.pars)

    def number_particles(self):
        """
        Returns
        -------
        int
            the total number of particles of the APR
        """
        return self.apr.nparticles()

    def get_intensities(self):
        """
        Returns
        -------
        numpy.ndarray
            the intensity values of the APR particles
        """
        return self.apr.get_intensities()

    def get_levels(self):
        """
        Returns
        -------
        numpy.ndarray
            the resolution levels of the APR particles
        """
        return self.apr.get_levels()

    def max_level(self):
        """
        Returns
        -------
        int
            the maximum resolution level of the APR
        """
        return self.apr.max_level()

    def min_level(self):
        """
        Returns
        -------
        int
            the minimum resolution level of the APR
        """
        return self.apr.min_level()

    def get_apr_from_array(self, arr):

        self.apr.get_apr_from_array(arr)


# TODO: implement this?
class APRTree:
    def __init__(self):
        self.apr_tree = None


# TODO: implement this?
class ParticleData:
    def __init__(self):
        self.data = None
