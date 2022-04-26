try:
    from ._version import version as __version__
except ImportError:
    __version__ = "not-installed"


from . import data_containers
from .data_containers import *
from .filegui import InteractiveIO
from . import converter
from . import io
from . import numerics
from . import viewer


try:
    from _pyaprwrapper import __cuda_build__
except ImportError:
    __cuda_build__ = False


def cuda_enabled():
    return __cuda_build__


__all__ = ['data_containers', 'io', 'viewer', 'converter', 'numerics']
