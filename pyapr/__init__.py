try:
    from ._version import version as __version__
except ImportError:
    __version__ = "not-installed"


from .data_containers import *
from .utils import InteractiveIO
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
from . import viewer


try:
    from _pyaprwrapper import __cuda_build__
except ImportError:
    __cuda_build__ = False


def cuda_enabled():
    return __cuda_build__
