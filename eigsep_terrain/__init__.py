__author__ = "Aaron Parsons"
__version__ = "0.0.1"

from . import dem
from . import ray
from . import utils
from . import plot

# img and seg need the optional `img` extra (torch, transformers, opencv,
# pymc). Import them eagerly only when it is installed, so that the terrain
# and horizon code stays usable on the base dependencies. Both are still
# importable directly, e.g. `from eigsep_terrain.img import HorizonImage`.
try:
    from . import img
    from . import seg
except ImportError:  # pragma: no cover - depends on what is installed
    pass
