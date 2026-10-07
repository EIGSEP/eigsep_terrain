"""Georeferencing and true-north regression tests, without external data."""
import hashlib
import numpy as np
import pytest
from PIL import Image, TiffImagePlugin
from pyproj import Geod
from eigsep_terrain.dem import DEM
from eigsep_terrain.marjum_dem import MarjumDEM


def tile(path, x=291000, y=4346000):
    tags = TiffImagePlugin.ImageFileDirectory_v2()
    tags[33922] = (0., 0., 0., float(x), float(y), 0.)
    tags[33550] = (0.5, 0.5, 0.)
    tags[34735] = (1, 1, 0, 2, 1025, 0, 1, 1, 3072, 0, 1, 6341)
    Image.fromarray(np.arange(16, dtype=np.float32).reshape(4, 4)).save(
        path, tiffinfo=tags)
    return str(path)


def test_pixel_centres_roundtrip_mosaic_and_cache(tmp_path):
    files = np.array([[tile(tmp_path/'sw.tif'),
                       tile(tmp_path/'nw.tif', y=4346002)],
                      [tile(tmp_path/'se.tif', x=291002),
                       tile(tmp_path/'ne.tif', x=291002, y=4346002)]])
    dem = DEM(cache_file=tmp_path/'dem.npz')
    dem.load_tif(files, survey_offset=[1, 2, 3])
    dem.map_crd = dict.fromkeys(('southbc', 'westbc', 'northbc', 'eastbc'), 0)
    np.testing.assert_equal(dem.raster_origin, [291000.25, 4345998.25])
    assert dem.data[0, 0] == 12
    assert dem.data[4, 4] == 12
    point = np.array([1.234, 2.345, 1600.123])
    lat, lon, alt = dem.enu_to_latlon(point)
    np.testing.assert_allclose(dem.latlon_to_enu(lat, lon, alt), point,
                               atol=1e-6)
    dem.save_cache()
    restored = DEM(cache_file=tmp_path/'dem.npz')
    np.testing.assert_allclose(restored.latlon_to_enu(lat, lon, alt), point,
                               atol=1e-6)
    np.testing.assert_equal(restored.data, dem.data)
    bad = np.array([[files[0, 0]], [tile(tmp_path/'bad.tif', x=291003)]])
    with pytest.raises(ValueError, match='contiguous'):
        dem.load_tif(bad)


def test_legacy_cache_requires_new_path_and_preserves_source(tmp_path):
    cache = tmp_path/'old.npz'
    np.savez(cache, dem=np.zeros((4, 4)), files=np.array([['old.tif']]))
    original_hash = hashlib.sha256(cache.read_bytes()).hexdigest()
    tif = tile(tmp_path/'tile.tif')
    xml = tmp_path/'tile.xml'
    xml.write_text('<metadata><idinfo><spdom><bounding>'
                   '<westbc>-113.4</westbc><eastbc>-113.3</eastbc>'
                   '<southbc>39.2</southbc><northbc>39.3</northbc>'
                   '</bounding></spdom></idinfo></metadata>')
    for clear_cache in (False, True):
        with pytest.raises(ValueError, match='build a new cache path'):
            MarjumDEM(cache_file=cache, clear_cache=clear_cache,
                      xml_file=xml, tif_files=np.array([[tif]]))
        assert hashlib.sha256(cache.read_bytes()).hexdigest() == original_hash

    new_cache = tmp_path/'new.npz'
    dem = MarjumDEM(cache_file=new_cache, xml_file=xml,
                    tif_files=np.array([[tif]]), survey_offset=[0, 0, 7])
    np.testing.assert_equal(dem.survey_offset, [0, 0, 7])
    with np.load(new_cache) as saved:
        assert int(saved['cache_version']) == 2
        np.testing.assert_equal(saved['files'], [[tif]])
    assert hashlib.sha256(cache.read_bytes()).hexdigest() == original_hash


def test_true_horizon_sign_from_independent_geodesic():
    dem = DEM()
    dem._set_projection(6341)
    dem.raster_origin = np.array(dem._forward.transform(-113.4024, 39.2489))
    dem.res = 1.
    dem.survey_offset = np.zeros(3)
    dem.data = np.full((160, 160), -100., dtype=np.float32)
    dem.data[130, 80] = 100.
    observer = np.array([80.1, 30.1, 0.])
    lat0, lon0, _ = dem.raster_to_latlon(observer)
    lat1, lon1, _ = dem.raster_to_latlon([80, 130, 100])
    az, _, _ = Geod(ellps='GRS80').inv(lon0, lat0, lon1, lat1)
    true, _ = dem.calc_horizon(*observer, n_az=1440)
    grid, _ = dem.calc_horizon(*observer, n_az=1440, azimuth_frame='grid')
    # Compare centre of nonzero peak support (pillar spans several bins).
    bins = np.arange(1440)*2*np.pi/1440
    peak = np.angle(np.mean(np.exp(1j*bins[true > 0])))
    assert abs(peak-np.deg2rad(az)) < np.deg2rad(.3)
    assert abs(dem.grid_convergence(*observer[:2])) > np.deg2rad(1.5)
    assert not np.array_equal(true, grid)


def test_true_ray_rotation_matches_grid_call(monkeypatch):
    import eigsep_terrain.dem as module
    dem = DEM()
    dem._set_projection(6341)
    dem.raster_origin = np.array(dem._forward.transform(-113.4024, 39.2489))
    dem.res = 1.
    dem.survey_offset = np.zeros(3)
    dem.data = np.zeros((10, 10), dtype=np.float32)
    captured = []

    def capture(E, N, U, start, rays, **kwargs):
        captured.append(rays.copy())
        return np.zeros(rays.shape[1])

    monkeypatch.setattr(module, 'ray_trace_basic', capture)
    start = np.array([4., 4., 2.])
    dem.ray_trace(start, 1, azimuth_frame='true')
    dem.ray_trace(start, 1, azimuth_frame='grid')
    gamma = dem.grid_convergence(*start[:2])
    grid_az = np.arctan2(captured[0][0], captured[0][1])
    true_az = np.arctan2(captured[1][0], captured[1][1])
    np.testing.assert_allclose(np.exp(1j*(grid_az+gamma)),
                               np.exp(1j*true_az), atol=1e-6)


def test_maxpool_preserves_partial_boundary_blocks():
    dem = DEM()
    dem.data = np.zeros((10, 10), dtype=np.float32)
    dem.data[-1, -1] = 123.
    assert dem.build_maxpool_pyramid()[-1][0].max() == 123.
    assert dem.data.shape == (10, 10)


def test_usgs_documented_tile_bounds():
    # Real USGS 12STJ9145 tiepoint/extent and XML bounds. This fixture
    # verifies the projection independently of a forward/inverse roundtrip.
    dem = DEM()
    dem._set_projection(6341)
    dem.raster_origin = np.array([291000.25, 4345000.25])
    dem.survey_offset = np.zeros(3)
    corners = [dem.raster_to_latlon([e, n, 0])
               for e in [-.25, 999.75] for n in [-.25, 999.75]]
    lat, lon, _ = np.asarray(corners).T
    np.testing.assert_allclose(
        [min(lon), max(lon), max(lat), min(lat)],
        [-113.4216185392, -113.4097333074, 39.2383940438, 39.2291508547],
        rtol=0, atol=1e-7)
