from _pyaprwrapper.data_containers import APR, APRParameters
from ..utils import InteractiveIO, type_to_particles
from _pyaprwrapper.converter import FloatConverter, ShortConverter, ByteConverter
import numpy as np
from typing import Union, Optional


__allowed_image_types__ = [np.uint8, np.uint16, np.float32]


def get_apr(image: np.ndarray,
            converter: Optional[Union[ByteConverter, ShortConverter, FloatConverter]] = None,
            params: Optional[APRParameters] = None,
            verbose: bool = False):
    """
    Convert an image to the APR.

    Parameters
    ----------
    image: numpy.ndarray
        Input pixel image as an array of 1-3 dimensions.
    converter: ByteConverter, ShortConverter or FloatConverter, optional
        Converter object used to compute the APR. By default, FloatConverter is used to avoid rounding errors in
        internal steps.
    params: APRParameters, optional
        If provided, sets parameters of the converter. If not, the parameters of the converter object is used, with
        'auto_parameters' set to True.
    verbose: bool
        Set the verbose mode of the converter (default: False).

    Returns
    -------
    apr: APR
        Generated APR object containing the adaptively sampled structure.
    parts: ByteParticles, ShortParticles or FloatParticles
        Sampled particle values (average downsampled from pixel values). The data type matches that of the input image
        (uint8, uint16 or float).
    """

    if not image.flags['C_CONTIGUOUS']:
        print('WARNING: \'image\' argument given to get_apr is not C-contiguous \n'
              'input image has been replaced with a C-contiguous copy of itself')
        image = np.ascontiguousarray(image)

    if image.dtype not in __allowed_image_types__:
        errstr = 'pyapr.converter.get_apr accepts images of type float32, uint16 and uint8, ' \
                 'but {} was given'.format(image.dtype)
        raise TypeError(errstr)

    apr = APR()
    converter = converter or FloatConverter()

    # set parameters
    if params is None:
        params = converter.get_parameters()
        params.auto_parameters = True

    converter.set_parameters(params)
    converter.verbose = verbose

    parts = type_to_particles(image.dtype)

    # compute the APR and sample particles
    converter.get_apr(apr, image)
    parts.sample_image(apr, image)

    if verbose:
        print('Total number of particles: {}'.format(apr.total_number_particles()))
        print('Compuatational Ratio: {}'.format(apr.computational_ratio()))

    return apr, parts


def get_apr_interactive(image: np.ndarray,
                        converter: Optional[Union[ByteConverter, ShortConverter, FloatConverter]] = None,
                        params: Optional[APRParameters] = None,
                        verbose: bool = False,
                        slider_decimals: int = 1):
    """
    Interactively convert an image array to the APR. The parameters `Ip_th`, `sigma_th` and `grad_th` are set
    with visual feedback.

    Parameters
    ----------
    image: numpy.ndarray
        Input (pixel) image as an array of 1-3 dimensions.
    converter: ByteConverter, ShortConverter or FloatConverter, optional
        Converter object used to compute the APR. By default, FloatConverter is used to avoid rounding errors in
        internal steps.
    params: APRParameters, optional
        If provided, sets parameters of the converter. Otherwise default parameters are used.
    verbose: bool
        Set the verbose mode of the converter (default: False).
    slider_decimals: int
        Number of decimals to use in the parameter sliders.

    Returns
    -------
    apr: APR
        Generated APR object containing the adaptively sampled structure.
    parts: ByteParticles, ShortParticles or FloatParticles
        Sampled particle values (average downsampled from pixel values). The data type matches that of the input image
        (uint8, uint16 or float).
    """

    if not image.flags['C_CONTIGUOUS']:
        print('WARNING: \'image\' argument given to get_apr_interactive is not C-contiguous \n'
              'input image has been replaced with a C-contiguous copy of itself')
        image = np.ascontiguousarray(image)

    if image.dtype not in __allowed_image_types__:
        errstr = 'pyapr.converter.get_apr_interactive accepts images of type float32, uint16 and uint8, ' \
                 'but {} was given'.format(image.dtype)
        raise TypeError(errstr)

    while image.ndim < 3:
        image = np.expand_dims(image, axis=0)

    # Initialize objects
    io_int = InteractiveIO()
    apr = APR()
    converter = converter or FloatConverter()

    if params is None:
        params = converter.get_parameters()
        params.auto_parameters = False

    converter.set_parameters(params)
    converter.verbose = verbose

    parts = type_to_particles(image.dtype)

    # launch interactive APR converter
    io_int.interactive_apr(converter, apr, image, slider_decimals=slider_decimals)

    if verbose:
        print('Total number of particles: {}'.format(apr.total_number_particles()))
        print('Compuatational Ratio: {}'.format(apr.computational_ratio()))

    # sample particles
    parts.sample_image(apr, image)

    return apr, parts


def find_parameters_interactive(image: np.ndarray,
                                converter: Optional[Union[ByteConverter, ShortConverter, FloatConverter]] = None,
                                params: Optional[APRParameters] = None,
                                verbose: bool = False,
                                slider_decimals: int = 1):
    """
    Interactively find the APR conversion parameters `Ip_th`, `sigma_th` and `grad_th` with visual feedback.

    Parameters
    ----------
    image: numpy.ndarray
        Input (pixel) image as an array of 1-3 dimensions.
    converter: ByteConverter, ShortConverter or FloatConverter, optional
        Converter object used to compute the APR. By default, FloatConverter is used to avoid rounding errors in
        internal steps.
    params: APRParameters, optional
        If provided, sets parameters of the converter. Otherwise default parameters are used.
    verbose: bool
        Set the verbose mode of the converter (default: False).
    slider_decimals: int
        Number of decimals to use in the parameter sliders.

    Returns
    -------
    params: APRParameters
        The parameters with the selected threshold values.
    """

    if not image.flags['C_CONTIGUOUS']:
        print('WARNING: \'image\' argument given to find_parameters_interactive is not C-contiguous \n'
              'input image has been replaced with a C-contiguous copy of itself')
        image = np.ascontiguousarray(image)

    if image.dtype not in __allowed_image_types__:
        errstr = 'pyapr.converter.find_parameters_interactive accepts input images of type float32, uint16 and uint8, ' \
                 'but {} was given'.format(image.dtype)
        raise TypeError(errstr)

    while image.ndim < 3:
        image = np.expand_dims(image, axis=0)

    # Initialize objects
    io_int = InteractiveIO()
    apr = APR()
    converter = converter or FloatConverter()

    if params is None:
        params = converter.get_parameters()
        params.auto_parameters = False

    converter.set_parameters(params)
    converter.verbose = verbose

    # launch the interactive parameter tool
    params = io_int.find_parameters_interactive(converter, apr, image, slider_decimals=slider_decimals)

    if verbose:
        print("---------------------------------")
        print("Using the following parameters:")
        print("grad_th = {}, sigma_th = {}, Ip_th = {}".format(params.grad_th, params.sigma_th, params.Ip_th))
        print("---------------------------------")

    return params
