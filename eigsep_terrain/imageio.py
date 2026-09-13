'''Uniform image loading (JPEG, PNG, HEIC, ...) with EXIF orientation
applied, shared by HorizonImage and the sky segmenter so both operate on
pixel arrays with the same rotation.'''

import numpy as np
from PIL import Image, ImageOps

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass


def load_image(filename):
    '''Load an image file as an (H, W, 3) uint8 RGB array, applying the
    EXIF Orientation tag (if any) so portrait/rotated phone photos come
    out right-side up.'''
    im = Image.open(filename)
    im = ImageOps.exif_transpose(im)
    return np.array(im.convert('RGB'))
