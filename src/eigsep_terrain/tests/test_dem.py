"""Tests for DEM.ray_trace with multiple backends."""
import numpy as np
import pytest
import healpy

from eigsep_terrain.dem import DEM


def _make_flat_dem(npx=100, res=0.5, altitude=0):
    """Minimal flat DEM: npx x npx grid, res m/px, uniform altitude."""
    dem = DEM()
    dem.res = res
    dem.data = np.full((npx, npx), altitude, dtype=np.int32)
    dem.map_crd = {
        'eastbc': 0, 'westbc': 0, 'northbc': 0, 'southbc': 0
    }
    dem.survey_offset = np.array([0, 0, 0])
    return dem


def test_backend_attribute_stored():
    dem = DEM(backend='jax')
    assert dem.backend == 'jax'
    assert DEM().backend == 'numpy'


@pytest.mark.parametrize("backend", ["numpy", "numba", "jax"])
def test_ray_trace_shape_and_horizon_filter(backend):
    if backend == "numba":
        pytest.importorskip("numba")
    if backend == "jax":
        pytest.importorskip("jax")

    dem = _make_flat_dem()
    nside = 8
    npix = healpy.nside2npix(nside)
    # Start well inside the DEM extent (50 m x 50 m) and above flat ground
    start_point = np.array([24.0, 24.0, 5.0], dtype=np.float32)

    r = dem.ray_trace(start_point, nside=nside, delta_r_m=0.5,
                      max_horizon_ang_deg=45, backend=backend)

    assert isinstance(r, np.ndarray)
    assert r.shape == (npix,)

    px = np.arange(npix)
    th, _ = healpy.pix2ang(nside, ipix=px)
    # HealPix theta is colatitude; elevation = pi/2 - theta.
    # Elevation > 45 deg  →  theta < pi/4  →  should be NaN
    above_45 = th < (np.pi / 2 - np.deg2rad(45))
    assert np.all(np.isnan(r[above_45])), \
        "All above-horizon rays must be NaN"

    # Near-nadir rays (elevation < -80 deg) must hit the flat ground
    near_nadir = th > (np.pi - np.deg2rad(10))
    assert np.all(np.isfinite(r[near_nadir])), \
        "Near-nadir rays must hit flat ground"


def test_ray_trace_backend_kwarg_overrides_instance():
    """backend= on the call overrides self.backend."""
    dem = _make_flat_dem()
    dem.backend = 'jax'          # instance default
    nside = 4
    start_point = np.array([24.0, 24.0, 5.0], dtype=np.float32)
    # Override to 'numpy' — must not import jax
    r = dem.ray_trace(start_point, nside=nside, backend='numpy')
    assert r.shape == (healpy.nside2npix(nside),)


def test_ray_trace_invalid_backend():
    dem = _make_flat_dem()
    start_point = np.array([24.0, 24.0, 5.0], dtype=np.float32)
    with pytest.raises(ValueError, match="Unknown backend"):
        dem.ray_trace(start_point, nside=4, backend='bogus')


@pytest.mark.parametrize("backend", ["numpy", "numba", "jax"])
def test_ray_trace_backends_agree(backend):
    """numpy, numba, and jax backends should return the same distances."""
    if backend == "numba":
        pytest.importorskip("numba")
    if backend == "jax":
        pytest.importorskip("jax")

    dem = _make_flat_dem()
    nside = 8
    start_point = np.array([24.0, 24.0, 5.0], dtype=np.float32)

    r_ref = dem.ray_trace(start_point, nside=nside, delta_r_m=0.5,
                          backend='numpy')
    r = dem.ray_trace(start_point, nside=nside, delta_r_m=0.5,
                      backend=backend)

    # NaN pattern must match
    np.testing.assert_array_equal(np.isnan(r_ref), np.isnan(r))
    # Finite values must agree within one step
    mask = np.isfinite(r_ref)
    np.testing.assert_allclose(r[mask], r_ref[mask], atol=0.5)
