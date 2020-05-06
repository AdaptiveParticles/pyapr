import pyapr


def get_apr_interactive(image, dtype='float', rel_error=0.1, gradient_smoothing=2, verbose=True):
    # Initialize objects
    io_int = pyapr.filegui.InteractiveIO()
    apr = pyapr.APR()
    par = pyapr.APRParameters()
    if dtype == 'float':
        parts = pyapr.FloatParticles()
        converter = pyapr.converter.FloatConverter()
    elif dtype == 'short':
        parts = pyapr.ShortParticles()
        converter = pyapr.converter.ShortConverter()
    elif dtype == 'byte':
        parts = pyapr.ByteParticles()
        converter = pyapr.converter.ByteConverter()
    else:
        raise Exception("get_apr_interactive dtype must be 'float', 'short' or 'byte' ")

    # Set some parameters
    par.auto_parameters = False
    par.rel_error = rel_error
    par.gradient_smoothing = gradient_smoothing
    converter.set_parameters(par)
    converter.set_verbose(verbose)

    # Compute APR and sample particle values

    io_int.interactive_apr(converter, apr, image)

    if verbose:
        print("Total number of particles: {} \n".format(apr.total_number_particles()))
        cr = image.size/apr.total_number_particles()
        print("Compuational Ratio: {:7.2f}".format(cr))
        print("Sampling particles ... \n")

    parts.sample_image(apr, image)

    return apr, parts
