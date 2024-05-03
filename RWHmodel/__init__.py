# Initialization file

"""RainWaterHavesting model (RWHmodel) is a hydrological model to determine the effectiveness of rainwater harvesting for diverse applications"""

from os.path import dirname, join, abspath

DATADIR = abspath(join(dirname(__file__), "data"))

__version__ = "0.1.0"

from .RWHmodel import *
