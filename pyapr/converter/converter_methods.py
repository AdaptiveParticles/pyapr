import pyapr
from pyapr.converter import FloatConverter, ShortConverter
import numpy as np
from typing import Union, Optional


__allowed_image_types__ = [np.uint8, np.uint16, np.float32]


def get_apr(image: np.ndarray,
            converter: Optional[Union[FloatConverter, ShortConverter]] = None,
            params: Optional[pyapr.APRParameters] = None,
            verbose: bool = False):
    """
    Convert an image array to the APR.

    Parameters
    ----------
    image: numpy.ndarray
        Input (pixel) image as an array of 1-3 dimensions.
    converter: FloatConverter or ShortConverter (optional)
        Converter object used to compute the APR. By default, FloatConverter is used to avoid rounding errors in
        internal steps.
    params: pyapr.APRParameters (optional)
        If provided, sets parameters of the converter. If not, the parameters of the converter object is used, with
        'auto_parameters' set to True.
    verbose: bool
        Set the verbose mode of the converter (default: False).

    Returns
    -------
    A tuple (apr, parts) where apr is the APR object and parts is the sampled particle data object of type
    FloatParticles or ShortParticles (depending on the input image data type).
    """

    if image.dtype not in __allowed_image_types__:
        errstr = 'pyapr.converter.get_apr accepts images of type float32, uint16 and uint8, ' \
                 'but {} was given'.format(image.dtype)
        raise TypeError(errstr)

    # initialize objects
    apr = pyapr.APR()
    converter = converter or pyapr.converter.FloatConverter()

    # set parameters
    if params is None:
        par = converter.get_parameters()
        par.auto_parameters = True
    else:
        par = params

    converter.set_parameters(par)
    converter.verbose = verbose

    if image.dtype == np.float32:
        parts = pyapr.FloatParticles()
    else:
        parts = pyapr.ShortParticles()

    # compute the APR and sample particles
    converter.get_apr(apr, image)
    parts.sample_image(apr, image)

    if verbose:
        print('Total number of particles: {}'.format(apr.total_number_particles()))
        print('Compuatational Ratio: {}'.format(apr.computational_ratio()))

    return apr, parts


def get_apr_interactive(image: np.ndarray,
                        converter: Optional[Union[FloatConverter, ShortConverter]] = None,
                        params: Optional[pyapr.APRParameters] = None,
                        verbose: bool = False,
                        slider_decimals: int = 1):
    """
    Interactively convert an image array to the APR. The parameters `Ip_th`, `sigma_th` and `grad_th` are set
    with visual feedback.

    Parameters
    ----------
    image: numpy.ndarray
        Input (pixel) image as an array of 1-3 dimensions.
    converter: FloatConverter or ShortConverter (optional)
        Converter object used to compute the APR. By default, FloatConverter is used to avoid rounding errors in
        internal steps.
    params: pyapr.APRParameters (optional)
        If provided, sets parameters of the converter. Otherwise default parameters are used.
    verbose: bool
        Set the verbose mode of the converter (default: False).
    slider_decimals: int
        Number of decimals to use in the parameter sliders.

    Returns
    -------
    A tuple (apr, parts) where apr is the APR object and parts is the sampled particle data object of type
    FloatParticles or ShortParticles (depending on the input image data type).
    """

    if image.dtype not in __allowed_image_types__:
        errstr = 'pyapr.converter.get_apr accepts images of type float32, uint16 and uint8, ' \
                 'but {} was given'.format(image.dtype)
        raise TypeError(errstr)

    while image.ndim < 3:
        image = np.expand_dims(image, axis=0)

    # Initialize objects
    io_int = pyapr.InteractiveIO()
    apr = pyapr.APR()
    converter = converter or pyapr.converter.FloatConverter()

    if params is None:
        par = converter.get_parameters()
        par.auto_parameters = False
    else:
        par = params

    converter.set_parameters(par)
    converter.verbose = verbose

    if image.dtype == np.float32:
        parts = pyapr.FloatParticles()
    else:
        parts = pyapr.ShortParticles()

    # launch interactive APR converter
    io_int.interactive_apr(converter, apr, image, slider_decimals=slider_decimals)

    if verbose:
        print('Total number of particles: {}'.format(apr.total_number_particles()))
        print('Compuatational Ratio: {}'.format(apr.computational_ratio()))

    # sample particles
    parts.sample_image(apr, image)

    return apr, parts


def find_parameters_interactive(image: np.ndarray,
                                converter: Optional[Union[FloatConverter, ShortConverter]] = None,
                                params: Optional[pyapr.APRParameters] = None,
                                verbose: bool = False,
                                slider_decimals: int = 1):
    """Same as `get_apr_interactive` but simply returns the selected parameters"""

    if image.dtype not in __allowed_image_types__:
        errstr = 'pyapr.converter.find_parameters_interactive accepts input images of type float32, uint16 and uint8, ' \
                 'but {} was given'.format(image.dtype)
        raise TypeError(errstr)

    while image.ndim < 3:
        image = np.expand_dims(image, axis=0)

    # Initialize objects
    io_int = pyapr.filegui.InteractiveIO()
    apr = pyapr.APR()
    converter = converter or pyapr.converter.FloatConverter()

    if params is None:
        par = converter.get_parameters()
        par.auto_parameters = False
    else:
        par = params

    converter.set_parameters(par)
    converter.verbose = verbose

    # launch interactive APR converter
    par = io_int.find_parameters_interactive(converter, apr, image, slider_decimals=slider_decimals)

    if verbose:
        print("---------------------------------")
        print("Using the following parameters:")
        print("grad_th = {}, sigma_th = {}, Ip_th = {}".format(par.grad_th, par.sigma_th, par.Ip_th))
        print("---------------------------------")

    return par
