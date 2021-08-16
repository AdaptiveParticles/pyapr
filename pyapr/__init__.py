"""Python wrappers for LibAPR

The main package of pyapr contains no features. These are instead available through the following subpackages:

Subpackages
-----------

data_containers
    fundamental data container classes

converter
    templated classes for creating APRs from images of different data types

io
    reading and writing APRs from/to file

numerics
    subpackage for processing using APRs

viewer
    a simple graphical user interface for visualizing results and exploring parameters
"""

from . import data_containers
from .data_containers import *
from .filegui import InteractiveIO
from . import converter
from . import io
from . import numerics
from . import viewer

__all__ = ['data_containers', 'io', 'viewer', 'converter', 'numerics', 'tests']
