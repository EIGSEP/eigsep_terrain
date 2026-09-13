__author__ = "Aaron Parsons"
__version__ = "0.0.1"

from . import imageio
from . import dem
from . import ray
from . import utils
from . import reflectivity
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
