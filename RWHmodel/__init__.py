# Initialization file

"""RainWaterHavesting model (RWHmodel) is a hydrological model to determine the effectiveness of rainwater harvesting for diverse applications."""

from os.path import abspath, dirname, join

DATADIR = abspath(join(dirname(__file__), "data"))

__version__ = "0.1.0"

from RWHmodel.model import *
from RWHmodel.reservoir import *
from RWHmodel.hydro_model import *
from RWHmodel.timeseries import *
from RWHmodel.utils import *
from RWHmodel.analysis import *
from RWHmodel.plot import *