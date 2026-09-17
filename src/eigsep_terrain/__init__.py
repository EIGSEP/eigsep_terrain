__author__ = "Aaron Parsons"
__version__ = "0.1.0"

from . import imageio
from . import dem
from . import ray
from . import utils
from . import img
from . import exif
from . import meta
from . import tiepoints
from . import fitio
from . import plot


def __getattr__(name):
    # seg.py pulls in torch + transformers, a real memory cost -- deferred
    # until something actually accesses et.seg, rather than paid by every
    # `import eigsep_terrain` (most uses of this package need pose
    # refinement, tie points, or antenna reprojection, none of which touch
    # segmentation once an image's img_seg_*.npz cache already exists).
    if name == 'seg':
        from . import seg
        return seg
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# img needs the optional `img` extra (torch, transformers, opencv,
# pymc). Import them eagerly only when it is installed, so that the terrain
# and horizon code stays usable on the base dependencies. Both are still
# importable directly, e.g. `from eigsep_terrain.img import HorizonImage`.
try:
    from . import img
except ImportError:  # pragma: no cover - depends on what is installed
    pass
