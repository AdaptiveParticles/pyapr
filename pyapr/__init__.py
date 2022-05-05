"""
Content-adaptive storage and processing of volumetric images in Python.

`pyapr` is a collection of tools and algorithms to convert images to and from the
Adaptive Particle Representation (APR), as well as to manipulate and view APR
images. The base namespace `pyapr` holds a number of data container classes
(see data_containers), while functions for generating, viewing and processing APR
images are imported via submodules:

converter
    Conversion from pixel images to APR.
data_containers
    Base classes used by the package.
filter
    Spatial convolution and filters for APR images.
io
    Reading and writing APR images.
measure
    Measurement of object properties, mainly using label images.
morphology
    Morphological operations, e.g. dilation and erosion, removing small objects or holes.
reconstruction
    Reconstruction of pixel values from APR images, in all or parts of the volume and at different resolutions.
restoration
    Restoration algorithms for APR images (currently only deconvolution).
segmentation
    Segmentation algorithms for APR images (currently only graphcut).
transform
    Transforms for APR images (currently only maximum projection)
tree
    Computing interior tree values from APR particles, used in many multi-resolution functions (e.g. viewers).
utils
    Utility functions for handling files and data types of APR classes.
viewer
    Visualization methods for APR images, e.g. slice viewer and raycast rendering.
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "not-installed"

from .data_containers import *
from . import converter
from . import filter
from . import io
from . import measure
from . import morphology
from . import reconstruction
from . import restoration
from . import segmentation
from . import transform
from . import tree
from . import utils
from . import viewer


try:
    from _pyaprwrapper import __cuda_build__
except ImportError:
    __cuda_build__ = False


def cuda_enabled() -> bool:
    """Returns True if pyapr was built with CUDA support, and False otherwise."""
    return __cuda_build__
