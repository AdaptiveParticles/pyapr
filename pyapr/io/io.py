from ..data_containers import *
import numpy as np


def from_file(fname, dtype=np.float32):
    """
    Load an APR from hdf5 file

    Parameters
    ----------
    fname : str
        path of the .h5 file to be loaded

    dtype : numpy.dtype
        data type of the stored APR (valid options: np.float32, np.uint16, np.uint8)

    Returns
    -------
    apr : APR
        an object of class APR
    """

    apr = APR(dtype=dtype)
    apr.apr.read_apr(fname)

    return apr


def to_file(apr, fname):
    """
    Write the APR to file as hdf5

    Parameters
    ----------
    apr : APR
        an object of class APR to be written to file

    fname : str
        name or path of the output .h5 file
    """

    if fname.split('.')[-1].lower() == 'h5':
        fname = fname.split('.')[:-1]

    apr.apr.write_apr(fname)


def reconstruct(apr, smooth=False, intensities=None, level_delta=0):
    """
    Reconstruct a pixel image from an APR and return the result as a numpy array.

    Parameters
    ----------
    apr : APR
        an APR.

    smooth : bool
        returns a smooth reconstruction if true, or a piecewise constant reconstruction otherwise (default).

    intensities : numpy.ndarray
        particle intensities to be used for the reconstruction. If None (default), the internal intensities are used.

    level_delta : int
        if intensities are provided for a downsampled version of the APR, specify the difference between the original
        maximum level and the actual maximum level with this parameter (default 0).

    Returns
    -------
    numpy.ndarray
        a numpy array containing the reconstructed pixel image
    """

    if not isinstance(apr, APR):
        raise TypeError('input must be an instance of class APR')

    if intensities == None:

        if smooth:
            tmp = apr.apr.reconstruct_smooth()
        else:
            tmp = apr.apr.reconstruct()

    else:

        tmp = apr.apr.recon_newints(intensities, level_delta)

    return np.array(tmp, copy=False)
