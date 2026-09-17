'''Extract GPS position, compass heading, and focal length from photo
EXIF metadata, and convert them into an initial HorizonImage camera-pose
guess so images don't have to be hand-tuned from scratch.'''

import numpy as np
from PIL import Image, ExifTags

from .imageio import load_image  # noqa: F401 (ensures HEIC opener is registered)
from .img import PRM_ORDER

GPS_IFD_TAG = 0x8825  # 34853, GPSInfo pointer in the base IFD
EXIF_IFD_TAG = 0x8769  # 34865, Exif sub-IFD pointer (image dims live here)

# GPS sub-IFD tag numbers (see EXIF 2.3 spec, tag group 0x8825)
GPS_LAT_REF, GPS_LAT = 1, 2
GPS_LON_REF, GPS_LON = 3, 4
GPS_ALT_REF, GPS_ALT = 5, 6
GPS_IMG_DIR_REF, GPS_IMG_DIR = 16, 17
GPS_H_POSITIONING_ERROR = 31  # meters, horizontal accuracy estimate

FOCAL_LENGTH_TAG = 37386  # mm, physical focal length
FOCAL_LENGTH_35MM_TAG = 41989  # mm, 35mm-film-equivalent focal length
DATETIME_ORIGINAL_TAG = 36867

FULL_FRAME_WIDTH_MM = 36.0  # width of a 35mm film frame, for f_px conversion


def _dms_to_decimal(dms, ref):
    '''Convert an EXIF (deg, min, sec) tuple + hemisphere ref to signed
    decimal degrees.'''
    deg, minute, sec = (float(x) for x in dms)
    dd = deg + minute / 60 + sec / 3600
    if ref in ('S', 'W'):
        dd = -dd
    return dd


def read_exif(filename):
    '''Read GPS position, compass heading, focal length, and timestamp
    from an image's EXIF metadata. Returns a dict with keys
    'lat', 'lon', 'alt', 'heading', 'focal_mm', 'focal_35mm',
    'datetime', each None if not present in the file.'''
    im = Image.open(filename)
    exif = im.getexif()
    out = dict(lat=None, lon=None, alt=None, heading=None,
               focal_mm=None, focal_35mm=None, datetime=None,
               h_err=None)

    exif_sub = exif.get_ifd(EXIF_IFD_TAG)
    if FOCAL_LENGTH_TAG in exif_sub:
        out['focal_mm'] = float(exif_sub[FOCAL_LENGTH_TAG])
    if FOCAL_LENGTH_35MM_TAG in exif_sub:
        out['focal_35mm'] = float(exif_sub[FOCAL_LENGTH_35MM_TAG])
    if DATETIME_ORIGINAL_TAG in exif_sub:
        out['datetime'] = exif_sub[DATETIME_ORIGINAL_TAG]

    gps = exif.get_ifd(GPS_IFD_TAG)
    if GPS_LAT in gps and GPS_LAT_REF in gps:
        out['lat'] = _dms_to_decimal(gps[GPS_LAT], gps[GPS_LAT_REF])
    if GPS_LON in gps and GPS_LON_REF in gps:
        out['lon'] = _dms_to_decimal(gps[GPS_LON], gps[GPS_LON_REF])
    if GPS_ALT in gps:
        alt = float(gps[GPS_ALT])
        if gps.get(GPS_ALT_REF, 0) == 1:
            alt = -alt
        out['alt'] = alt
    if GPS_IMG_DIR in gps:
        out['heading'] = float(gps[GPS_IMG_DIR])  # deg, compass bearing
    if GPS_H_POSITIONING_ERROR in gps:
        out['h_err'] = float(gps[GPS_H_POSITIONING_ERROR])  # meters
    return out


def initial_pose_from_exif(exif, dem, npix_x, npix_y):
    '''Convert an exif dict (from read_exif) into an initial camera-pose
    guess ordered per PRM_ORDER = (e, n, u, th, ph, ti, f), using dem to
    convert lat/lon/alt to local ENU meters.

    th (elevation) and ti (roll) default to a level, unrotated shot
    (th=pi/2, ti=0); only ph (azimuth) and f (focal length in px) are
    informed by EXIF beyond position. Refine the rest with a local
    optimization against the DEM horizon (see Phase 2) before trusting
    it for MCMC.'''
    if exif.get('lat') is None or exif.get('lon') is None:
        raise ValueError('exif dict has no GPS lat/lon; cannot place camera.')
    e, n, u = dem.latlon_to_enu(exif['lat'], exif['lon'], exif.get('alt'))

    if exif.get('heading') is not None:
        # GPSImgDirection is a compass bearing (0=N, 90=E, clockwise).
        # HorizonImage.get_rays applies ph as an ordinary z-axis rotation
        # to a boresight that starts along +E (see rot_m in utils.py), i.e.
        # a math-convention angle (0=E, counterclockwise). The two are
        # related by ph = pi/2 - heading.
        ph = np.pi / 2 - np.deg2rad(exif['heading'])
    else:
        ph = 0.0

    th = np.pi / 2  # assume a level, horizon-pointing shot
    ti = 0.0  # assume no in-plane roll

    if exif.get('focal_35mm') is not None:
        f = npix_x * (exif['focal_35mm'] / FULL_FRAME_WIDTH_MM)
    else:
        f = float(npix_x)  # crude fallback: roughly a 1x (36mm-equiv) lens

    prms = dict(e=e, n=n, u=u, th=th, ph=ph, ti=ti, f=f)
    return np.array([prms[k] for k in PRM_ORDER], dtype=np.float32)
