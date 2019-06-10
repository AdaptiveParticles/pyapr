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

nn
    pytorch modules for APR based neural networks

numerics
    subpackage for processing using APRs

viewer
    a simple graphical user interface for visualizing results and exploring parameters
"""

from .data_containers import *
from .converter import *
from .io import *
from .numerics import *
#from .viewer import *

__all__ = ['data_containers', 'io', 'nn', 'viewer', 'converter', 'numerics']
