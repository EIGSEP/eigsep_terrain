"""Tests for exif.py's pure conversion logic (no real image files)."""
import numpy as np
import pytest

from eigsep_terrain.exif import _dms_to_decimal, initial_pose_from_exif
from eigsep_terrain.img import PRM_ORDER


def test_dms_to_decimal_north_east_positive():
    assert _dms_to_decimal((39, 14, 52.5), 'N') == pytest.approx(39.24791667, abs=1e-6)


def test_dms_to_decimal_south_west_negative():
    lat = _dms_to_decimal((39, 14, 52.5), 'S')
    lon = _dms_to_decimal((113, 24, 9.5), 'W')
    assert lat < 0
    assert lon < 0


class _FakeDEM:
    def latlon_to_enu(self, lat, lon, alt=None):
        return np.array([1.0, 2.0, 3.0 if alt is None else alt])


def test_initial_pose_from_exif_orders_prms_correctly():
    exif = dict(lat=39.25, lon=-113.40, alt=1800.0,
                heading=90.0, focal_35mm=26.0)
    prms = initial_pose_from_exif(exif, _FakeDEM(), npix_x=4032, npix_y=3024)
    assert prms.shape == (len(PRM_ORDER),)
    d = dict(zip(PRM_ORDER, prms))
    assert d['e'] == pytest.approx(1.0)
    assert d['n'] == pytest.approx(2.0)
    assert d['u'] == pytest.approx(1800.0)
    # heading=90 (due east) -> ph = pi/2 - pi/2 = 0
    assert d['ph'] == pytest.approx(0.0, abs=1e-6)
    assert d['th'] == pytest.approx(np.pi / 2)
    assert d['ti'] == pytest.approx(0.0)
    assert d['f'] == pytest.approx(4032 * 26.0 / 36.0)


def test_initial_pose_from_exif_missing_heading_defaults_zero():
    exif = dict(lat=39.25, lon=-113.40, alt=1800.0,
                heading=None, focal_35mm=None)
    prms = initial_pose_from_exif(exif, _FakeDEM(), npix_x=4032, npix_y=3024)
    d = dict(zip(PRM_ORDER, prms))
    assert d['ph'] == 0.0
    assert d['f'] == pytest.approx(4032.0)


def test_initial_pose_from_exif_requires_gps():
    with pytest.raises(ValueError):
        initial_pose_from_exif(dict(lat=None, lon=None), _FakeDEM(), 100, 100)
