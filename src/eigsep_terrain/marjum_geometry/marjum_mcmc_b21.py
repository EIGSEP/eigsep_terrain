"""Antenna posterior for the marjum-2026-07 91 m era, on the current 23-camera state.

Why this exists separately from ``marjum_mcmc``
-----------------------------------------------
``marjum_mcmc`` and its two pilot runs sample a 13-camera network taken from
``cv_initialization_v3``, against a ``meta.json`` whose hash no longer matches
the working copy (it predates the antenna repick) and with no lens distortion.
The current photogrammetry product is ``cv_antenna_repick_v1/fit_antenna.npz``:
23 cameras, per-lens-group radial distortion, 3,200 landmarks, 11,261 shared
feature observations.  The two networks disagree by up to 22 m per camera, so
the pilots answer a superseded question.  ``marjum_mcmc`` is left untouched
because existing run manifests record its SHA-256.

What changed in the sampler
---------------------------
The pilots moved all 97 global coordinates in one Metropolis block.  That block
is the expensive one: a global move rewrites every camera pose and therefore
every cached skyline, which is ~95% of the cost of a density evaluation.  It
also mixes the worst, because a 97-dimensional random walk explores slowly and
the target is rugged.

This module blocks by physical coupling instead:

* one block per camera (7 coordinates) -- only that camera's horizon, its own
  feature rows, and its own antenna label change, so a camera move costs 1/23
  of a full skyline;
* one antenna block (3 coordinates) -- the antenna appears in no horizon and no
  feature row, so an antenna move costs 23 point projections and nothing else;
* one nuisance block (GPS bias, antenna scatter);
* conditionally independent landmark moves, as in the pilot, which already
  mixed at ~35% acceptance;
* an occasional whole-network translation, which is the move that addresses the
  weak horizontal direction identified in the view geometry (cameras, antenna
  and landmarks shift together; only the DEM-referenced terms resist).

The target density is the same functional form as the pilot's and is evaluated
exactly -- no Laplace approximation and no landmark marginalization.  Blocking
changes only which coordinates move together, not the stationary distribution.

Cameras IMG_2209 and IMG_2210 carry no EXIF GPS fix and no heading.  Their
position, altitude and heading priors are dropped rather than invented; they
are constrained by shared features and their own horizons.  These are two of
the four opposite-side views that carry most of the horizontal information, so
this choice matters and is recorded in the manifest.
"""
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import argparse
import hashlib
import json
import os
import pickle
import sys
import time
import types

import numpy as np
from eigsep_terrain import exif as terrain_exif


def _install_fitio_shim():
    """Restore ``eigsep_terrain.fitio`` for modules that still import it.

    The installed package dropped the module in the src/ layout restructure;
    ``marjum_bundle`` and ``marjum_mcmc`` import it at call time.  Injecting the
    local implementation keeps those files byte-identical, which keeps the
    SHA-256 entries in existing run manifests valid.
    """
    import eigsep_terrain
    import marjum_fitio
    if 'eigsep_terrain.fitio' in sys.modules:
        return
    shim = types.ModuleType('eigsep_terrain.fitio')
    shim.load_fit, shim.save_fit = marjum_fitio.load_fit, marjum_fitio.save_fit
    sys.modules['eigsep_terrain.fitio'] = shim
    eigsep_terrain.fitio = shim


_install_fitio_shim()

from marjum_bundle import Terrain, PRM_ORDER  # noqa: E402
from marjum_camera import project, rays  # noqa: E402
from marjum_mcmc import student_residual, digest, focal_prior_widths  # noqa: E402

STATE_FILE = 'cv_antenna_repick_v1/fit_antenna.npz'
DEM_FILE = 'marjum_dem_sw.npz'

# The published immutable geometry release this run samples against. The
# release pins the SHA-256 of the antenna fit used as the state file, so a run
# can assert it is sampling exactly the released geometry rather than a working
# copy that has since drifted.
RELEASE_ID = 'v0001_marjum_geometry'
RELEASE_MANIFEST = Path('v0001_marjum_geometry_snapshot/manifest.json')


def release_check(state_file, manifest_path=RELEASE_MANIFEST):
    """Verify the state file matches the hash pinned by the geometry release.

    Returns a dict recorded verbatim in the run manifest. Raises if the release
    pins a different hash, because sampling a state that is not the released
    one would silently produce a posterior nobody can trace back to a release.
    """
    state_file = Path(state_file)
    actual = digest(state_file)
    info = dict(release=RELEASE_ID, manifest=str(manifest_path),
                state_file=str(state_file), state_sha256=actual)
    if not manifest_path.exists():
        info.update(verified=False, reason='release manifest not present in working tree')
        return info
    release = json.loads(manifest_path.read_text())
    info['release_generated_utc'] = release.get('generated_utc')
    info['release_status'] = release.get('status')
    pinned = {a['path']: a['sha256'] for a in release.get('source_artifacts', [])}
    want = pinned.get(f'terrain/{state_file}')
    info['pinned_sha256'] = want
    if want is None:
        info.update(verified=False, reason=f'{state_file} is not pinned by {RELEASE_ID}')
        return info
    if want != actual:
        raise SystemExit(
            f'state file {state_file} does not match {RELEASE_ID}\n'
            f'  pinned {want}\n  actual {actual}\n'
            'Refusing to sample: the posterior would not correspond to the released geometry.')
    info.update(verified=True)
    # Report, but do not fail on, drift in the other artifacts the release
    # pins. marjum_bundle.py is expected to differ: working_grid was restored
    # to it on 2026-09-14, after the release was cut, and this module requires
    # that function (see the note in Posterior.__init__). meta.json is also
    # expected to differ: 2211's transmitter pixel was re-picked after the
    # release, which does not affect this antenna-era posterior.
    drift = {}
    for path, want_hash in pinned.items():
        local = Path(path.split('terrain/', 1)[-1])
        if local == state_file or not local.exists():
            continue
        have = digest(local)
        if have != want_hash:
            drift[str(local)] = dict(pinned=want_hash, actual=have)
    info['other_pinned_artifacts_that_differ'] = drift
    return info


@dataclass(frozen=True)
class Config:
    """Prior and likelihood scales.

    Identical to ``marjum_mcmc.Config`` except for ``skyline_samples``.  The
    pilot used 768; comparing 768/3072/12288 on the same state gives 5.5 px RMS
    between the first two and 1.9 px RMS between the last two, against an
    assumed 20 px horizon scale.  4096 keeps the surrogate's own discretization
    error small compared with the modelled scale without making a camera move
    unaffordable.  It is still a sampled-ray surrogate, not a raster horizon.
    """
    tie_sigma_px: float = 4.
    antenna_label_sigma_px: float = 3.
    antenna_extra_prior_px: float = 10.
    horizon_sigma_angular_px: float = 20.
    terrain_sigma_m: float = 5.
    gps_independent_floor_m: float = 10.
    gps_common_sigma_m: float = 20.
    altitude_sigma_m: float = 30.
    heading_sigma_rad: float = .5
    elevation_sigma_rad: float = .5
    roll_sigma_rad: float = .2
    log_f_sigma: float = .25
    log_f_sigma_by_camera: dict | None = None
    student_df: float = 4.
    skyline_samples: int = 4096


class Posterior:
    """Joint posterior over 23 camera poses, the antenna, landmarks and nuisances.

    Coordinate layout of the global vector (``self.ng`` entries)::

        [camera 0..22] x (e, n, u, th, ph, ti, log f)   161
        antenna (e, n, u)                                 3
        common GPS bias (e, n)                            2
        excess antenna label scatter (px)                 1

    Landmarks follow as ``3 * n_points`` further coordinates.  Camera focal
    length is sampled as its logarithm; everything else is sampled directly in
    metres, radians or pixels, so there is no scaling to undo when reporting.
    """

    def __init__(self, state_file=STATE_FILE, dem_file=DEM_FILE, meta_file='meta.json',
                 exif_file='marjum_2026_07_exif.npz', config=None, feature_dir='cv_features'):
        from eigsep_terrain.marjum_dem import MarjumDEM
        self.config = config or Config()
        self.state_file = Path(state_file)
        self.dem_file = dem_file
        self.feature_dir = Path(feature_dir)
        # Terrain.__init__ calls working_grid, which restores the anchor that
        # MarjumDEM.load_cache discards for the _sw mosaic.  A raw MarjumDEM
        # here would place every camera ~255 m underground.
        self.terrain = Terrain(MarjumDEM(cache_file=dem_file))

        with np.load(self.state_file, allow_pickle=True) as s:
            self.keys = [str(k) for k in s['keys']]
            self.cameras0 = np.asarray(s['cameras'], float)
            self.distortion = np.asarray(s['distortion'], float)
            self.groups = np.asarray(s['groups'], int)
            self.shapes = np.asarray(s['shapes'], int)
            self.points0 = np.asarray(s['points'], float)
            self.antenna0 = np.asarray(s['antenna'], float)
            self.oc = np.asarray(s['obs_cam'], int)
            self.op = np.asarray(s['obs_point'], int)
            self.xy = np.asarray(s['obs_xy'], float)
            # Joint mode: sample the transmitter alongside the antenna whenever
            # the state carries one. An antenna-only state has none, and the
            # coordinate layout below then drops those three coordinates.
            self.transmitter0 = np.asarray(s['transmitter'], float) if 'transmitter' in s.files else None
        self.nc = len(self.keys)
        self.npoint = int(self.op.max()) + 1
        # 7 per camera, antenna 3, transmitter 3 (joint only), GPS bias 2, scatter 1.
        self.ntx = 3 if self.transmitter0 is not None else 0
        # Joint runs also estimate an excess scatter for the transmitter picks,
        # exactly as the antenna does. Without it a 3 px label sigma would make
        # the six ultra-close conditioned cameras overwhelm 2210/2211 and hide
        # the known group disagreement behind a spuriously tight posterior;
        # with it, the model can report how inconsistent those picks really are.
        self.ntxe = 1 if self.ntx else 0
        self.ng = self.nc*7 + 6 + self.ntx + self.ntxe
        self.ndim = self.ng + 3*self.npoint
        self.by_camera = [np.flatnonzero(self.oc == i) for i in range(self.nc)]
        order = np.argsort(self.op, kind='stable')
        bounds = np.searchsorted(self.op[order], np.arange(self.npoint + 1))
        self.by_point = [order[bounds[j]:bounds[j+1]] for j in range(self.npoint)]

        self.joint = self.transmitter0 is not None

        meta = json.loads(Path(meta_file).read_text())
        # Labels are per-target and not every camera carries both: in the 29-pose
        # joint set, 23 have an antenna pick and 8 have a transmitter pick (the
        # six transmitter-conditioned cameras plus 2210/2211, which have both).
        # An antenna-only state must still have a label for every camera.
        self.axy = np.array([meta.get(k, {}).get('ant_px', [np.nan, np.nan]) for k in self.keys], float)
        self.has_ant_label = np.isfinite(self.axy).all(axis=1)
        self.txy = np.array([meta.get(k, {}).get('transmitter_px', [np.nan, np.nan])
                             for k in self.keys], float)
        self.has_tx_label = np.isfinite(self.txy).all(axis=1)
        if not self.joint and not self.has_ant_label.all():
            missing = [k for k, ok in zip(self.keys, self.has_ant_label) if not ok]
            raise ValueError(f'No ant_px label for {missing}')
        if self.joint and not self.has_tx_label.any():
            raise ValueError('state carries a transmitter but no image has a transmitter_px label')
        self.n_ant_label = int(self.has_ant_label.sum())
        self.n_tx_label = int(self.has_tx_label.sum())

        self.horizon = []
        code_dir = Path(__file__).resolve().parent
        self.input_files = [Path(__file__), Path(terrain_exif.__file__), code_dir/'marjum_fitio.py', code_dir/'marjum_camera.py',
                            code_dir/'marjum_bundle.py', code_dir/'marjum_mcmc.py', Path(meta_file),
                            self.state_file, Path(dem_file), Path(exif_file)]
        for k in self.keys:
            file = self.feature_dir / f'sift_{k}.npz'
            with np.load(file) as f:
                self.horizon.append(np.asarray(f['horizon'], float))
            self.input_files.append(file)

        with np.load(exif_file) as ex:
            names = [str(k) for k in ex['keys']]
            idx = [names.index(k) for k in self.keys]
            self.gps = np.c_[ex['e_gps'][idx], ex['n_gps'][idx]]
            self.gps_sigma = np.maximum(ex['h_err_m'][idx], self.config.gps_independent_floor_m)
            self.alt = np.asarray(ex['u_gps'][idx], float)
            self.heading = np.pi/2 - np.deg2rad(ex['heading_deg'][idx])
            self.focal = terrain_exif.focal_length_pixels(ex['focal_35mm'][idx],
                                                          self.shapes[:,1], self.shapes[:,0])
        self.focal_sigma = focal_prior_widths(self.config, self.keys)
        # IMG_2209 and IMG_2210 have has_gps False: no fix, no heading. Use the
        # feature and horizon terms for them instead of a fabricated prior.
        self.has_gps = np.isfinite(self.gps).all(axis=1) & np.isfinite(self.gps_sigma)
        self.has_heading = np.isfinite(self.heading)
        self.has_alt = np.isfinite(self.alt)
        if not np.all(np.isfinite(self.focal)):
            raise ValueError('EXIF focal length missing; it is required for every view')
        self.unpriored = [k for k, g in zip(self.keys, self.has_gps) if not g]

    # ---------------------------------------------------------------- layout

    def unpack(self, z):
        z = np.asarray(z, float)
        cam = z[:7*self.nc].reshape(self.nc, 7).copy()
        with np.errstate(over='ignore'):
            cam[:, 6] = np.exp(cam[:, 6])
        a = 7*self.nc
        tx = z[a+3:a+3+self.ntx] if self.ntx else None
        b = a + 3 + self.ntx
        tx_extra = float(z[b+3]) if self.ntxe else None
        return (cam, z[a:a+3], tx, z[b:b+2], z[b+2], tx_extra,
                z[self.ng:].reshape(self.npoint, 3))

    def pack(self, cam, ant, tx, bias, extra, tx_extra, points):
        cam = np.asarray(cam, float).copy()
        cam[:, 6] = np.log(cam[:, 6])
        parts = [cam.ravel(), np.asarray(ant, float)]
        if self.ntx:
            parts.append(np.asarray(tx, float))
        parts += [np.asarray(bias, float), [extra]]
        if self.ntxe:
            parts.append([tx_extra])
        parts.append(np.asarray(points, float).ravel())
        return np.concatenate([np.atleast_1d(p) for p in parts])

    def start_vector(self):
        cameras = self.cameras0.copy()
        # The saved azimuth and the EXIF heading can differ by a full turn. The
        # prior lives on one branch, so fold the start onto it; this is a
        # relabelling of the same pose, not a change of pose.
        wrapped = self.heading + (cameras[:, 4] - self.heading + np.pi) % (2*np.pi) - np.pi
        cameras[:, 4] = np.where(self.has_heading, wrapped, cameras[:, 4])
        return self.pack(cameras, self.antenna0, self.transmitter0, np.zeros(2), 8.,
                         (20. if self.ntxe else None), self.points0)

    # ------------------------------------------------------------ components

    def _camera_support(self, cam_i):
        t = self.terrain
        if not np.all(np.isfinite(cam_i)):
            return False
        if not (0 < cam_i[3] < np.pi) or abs(cam_i[5]) >= np.pi:
            return False
        if not (100 < cam_i[6] < 50000):
            return False
        if not (t.e[0]+2 < cam_i[0] < t.e[-1]-2 and t.n[0]+2 < cam_i[1] < t.n[-1]-2):
            return False
        return bool(cam_i[2] > t.height(cam_i[0], cam_i[1]) + .1)

    def camera_logp(self, i, cam_i, ant, points, bias, extra, horizon=None, tx=None,
                    tx_extra=0.):
        """Every term that changes when camera ``i`` alone moves.

        ``horizon`` may carry an already-computed ``horizon_logp(i, cam_i)`` for
        this exact pose.  The horizon term depends on nothing but the camera, so
        it survives antenna, landmark and nuisance updates and only has to be
        recomputed when the camera itself moves.  That caching is what keeps a
        sweep at one skyline pass rather than two.
        """
        residual = self.camera_residuals(i, cam_i, ant, points, bias, extra,
                                         tx=tx, tx_extra=tx_extra)
        if residual is None:
            return -np.inf
        total = -.5*np.dot(residual, residual)
        total += self.horizon_logp(i, cam_i) if horizon is None else horizon
        return float(total)

    def camera_residuals(self, i, cam_i, ant, points, bias, extra, tx=None,
                         tx_extra=0.):
        """Camera-dependent residuals except the separately cached horizon.

        Squared norms reproduce ``camera_logp``. These same residuals supply
        Gauss-Newton proposal curvature, never an approximate acceptance ratio.
        ``None`` denotes a pose outside the target's support.
        """
        if not self._camera_support(cam_i):
            return None
        c = self.config
        k = self.distortion[i]
        shape = self.shapes[i]

        idx = self.by_camera[i]
        pred, depth = project(cam_i, shape, points[self.op[idx]], k)
        if np.any(depth <= .1):
            return None
        tie = student_residual((pred - self.xy[idx])/c.tie_sigma_px, c.student_df, 2)
        residual = [tie.ravel()]

        sigma = np.hypot(c.antenna_label_sigma_px, extra)
        if self.has_ant_label[i]:
            ap, adepth = project(cam_i, shape, ant, k)
            if adepth[0] <= .1:
                return None
            residual.append((ap[0] - self.axy[i])/sigma)

        # Transmitter pick, on the eight views that carry one. Manual picks on
        # the transmitter structure have their own excess-scatter term.
        if tx is not None and self.has_tx_label[i]:
            tp, tdepth = project(cam_i, shape, tx, k)
            if tdepth[0] <= .1:
                return None
            residual.append((tp[0] - self.txy[i])/np.hypot(c.antenna_label_sigma_px, tx_extra))

        if self.has_gps[i]:
            residual.append((cam_i[:2] + bias - self.gps[i])/self.gps_sigma[i])
        if self.has_alt[i]:
            residual.append([(cam_i[2] - self.alt[i])/c.altitude_sigma_m])
        if self.has_heading[i]:
            delta = cam_i[4] - self.heading[i]
            if abs(delta) >= np.pi:
                return None
            residual.append([delta/c.heading_sigma_rad])
        residual.append([(cam_i[3] - np.pi/2)/c.elevation_sigma_rad,
                         cam_i[5]/c.roll_sigma_rad,
                         np.log(cam_i[6]/self.focal[i])/self.focal_sigma[i]])
        return np.concatenate(residual)

    def horizon_logp(self, i, cam_i):
        r = self.horizon_residuals(i, cam_i)
        return float(-.5*np.sum(r*r))

    def horizon_residuals(self, i, cam_i):
        c = self.config
        d = rays(cam_i, self.shapes[i], self.horizon[i], self.distortion[i])
        elevation = np.arctan2(d[:, 2], np.hypot(d[:, 0], d[:, 1]))
        skyline = self.terrain.skyline(cam_i[:3], np.arctan2(d[:, 1], d[:, 0]),
                                       c.skyline_samples)
        # Angular scale fixed by the EXIF focal length, not the sampled one, so
        # the horizon weight cannot be traded against focal length.
        return student_residual((elevation - skyline)*self.focal[i]/c.horizon_sigma_angular_px,
                                c.student_df)

    def antenna_logp(self, cam, ant, extra):
        """Every term that changes when the antenna alone moves."""
        t = self.terrain
        if not np.all(np.isfinite(ant)):
            return -np.inf
        if not (t.e[0]+2 < ant[0] < t.e[-1]-2 and t.n[0]+2 < ant[1] < t.n[-1]-2):
            return -np.inf
        if not (float(t.height(ant[0], ant[1])) < ant[2] < 4000.):
            return -np.inf
        c = self.config
        sigma = np.hypot(c.antenna_label_sigma_px, extra)
        total = 0.
        for i in np.flatnonzero(self.has_ant_label):
            ap, depth = project(cam[i], self.shapes[i], ant, self.distortion[i])
            if depth[0] <= .1:
                return -np.inf
            total -= .5*np.sum(((ap[0] - self.axy[i])/sigma)**2)
        return float(total)

    def transmitter_logp(self, cam, tx, tx_extra=0.):
        """Every term that changes when the transmitter alone moves.

        Only the views carrying a transmitter pick contribute. Six of them are
        the transmitter-conditioned cameras, whose own poses are fitted against
        this same point, so their agreement is not independent evidence; 2210
        and 2211 are the two views that are not conditioned on it.
        """
        t = self.terrain
        if not np.all(np.isfinite(tx)):
            return -np.inf
        if not (t.e[0]+2 < tx[0] < t.e[-1]-2 and t.n[0]+2 < tx[1] < t.n[-1]-2):
            return -np.inf
        if not (float(t.height(tx[0], tx[1])) - 5. < tx[2] < 4000.):
            return -np.inf
        c = self.config
        sigma = np.hypot(c.antenna_label_sigma_px, tx_extra)
        total = 0.
        for i in np.flatnonzero(self.has_tx_label):
            tp, depth = project(cam[i], self.shapes[i], tx, self.distortion[i])
            if depth[0] <= .1:
                return -np.inf
            total -= .5*np.sum(((tp[0] - self.txy[i])/sigma)**2)
        return float(total)

    def tx_scatter_logp(self, cam, tx, tx_extra):
        """Transmitter-label terms plus the scatter-dependent normalization."""
        if tx_extra < 0:
            return -np.inf
        c = self.config
        total = self.transmitter_logp(cam, tx, tx_extra)
        if not np.isfinite(total):
            return -np.inf
        sigma = np.hypot(c.antenna_label_sigma_px, tx_extra)
        total -= self.n_tx_label*2*np.log(sigma/c.antenna_label_sigma_px)
        total -= .5*(tx_extra/c.antenna_extra_prior_px)**2
        return float(total)

    def scatter_logp(self, cam, ant, extra):
        """Antenna-label terms plus the scatter-dependent Gaussian normalization."""
        if extra < 0:
            return -np.inf
        c = self.config
        sigma = np.hypot(c.antenna_label_sigma_px, extra)
        total = self.antenna_logp(cam, ant, extra)
        if not np.isfinite(total):
            return -np.inf
        # Two-dimensional Gaussian normalization per antenna-labelled view.
        total -= self.n_ant_label*2*np.log(sigma/c.antenna_label_sigma_px)
        total -= .5*(extra/c.antenna_extra_prior_px)**2
        return float(total)

    def point_logp(self, z=None, cam=None, points=None):
        """One independent conditional log density per landmark."""
        if cam is None:
            cam, _, _, _, _, _, points = self.unpack(z)
        c = self.config
        pred = np.empty_like(self.xy)
        depth = np.empty(len(self.xy))
        for i, idx in enumerate(self.by_camera):
            pred[idx], depth[idx] = project(cam[i], self.shapes[i], points[self.op[idx]],
                                            self.distortion[i])
        tie = student_residual((pred - self.xy)/c.tie_sigma_px, c.student_df, 2)
        height = self.terrain.height(points[:, 0], points[:, 1])
        terrain = student_residual((points[:, 2] - height)/c.terrain_sigma_m, c.student_df)
        value = -.5*(np.bincount(self.op, weights=np.sum(tie*tie, axis=1), minlength=self.npoint)
                     + terrain*terrain)
        invalid = np.bincount(self.op, weights=(depth <= .1), minlength=self.npoint) > 0
        t = self.terrain
        invalid |= ((points[:, 0] <= t.e[0]+2) | (points[:, 0] >= t.e[-1]-2) |
                    (points[:, 1] <= t.n[0]+2) | (points[:, 1] >= t.n[-1]-2))
        value[invalid | ~np.isfinite(value)] = -np.inf
        return value

    def logp(self, z):
        """Exact joint log density, used for checks and for whole-network moves."""
        cam, ant, tx, bias, extra, tx_extra, points = self.unpack(z)
        if extra < 0 or (tx_extra is not None and tx_extra < 0):
            return -np.inf
        point_part = self.point_logp(cam=cam, points=points)
        if not np.all(np.isfinite(point_part)):
            return -np.inf
        c = self.config
        total = float(np.sum(point_part))
        sigma = np.hypot(c.antenna_label_sigma_px, extra)
        t = self.terrain
        if not np.all(np.isfinite(ant)):
            return -np.inf
        if not (t.e[0]+2 < ant[0] < t.e[-1]-2 and t.n[0]+2 < ant[1] < t.n[-1]-2):
            return -np.inf
        if not (float(t.height(ant[0], ant[1])) < ant[2] < 4000.):
            return -np.inf
        if tx is not None:
            tx_part = self.tx_scatter_logp(cam, tx, tx_extra)
            if not np.isfinite(tx_part):
                return -np.inf
            total += tx_part
        for i in range(self.nc):
            if not self._camera_support(cam[i]):
                return -np.inf
            if self.has_ant_label[i]:
                ap, depth = project(cam[i], self.shapes[i], ant, self.distortion[i])
                if depth[0] <= .1:
                    return -np.inf
                total -= .5*np.sum(((ap[0] - self.axy[i])/sigma)**2)
            total += self.horizon_logp(i, cam[i])
            if self.has_gps[i]:
                total -= .5*np.sum(((cam[i, :2] + bias - self.gps[i])/self.gps_sigma[i])**2)
            if self.has_alt[i]:
                total -= .5*((cam[i, 2] - self.alt[i])/c.altitude_sigma_m)**2
            if self.has_heading[i]:
                delta = cam[i, 4] - self.heading[i]
                if abs(delta) >= np.pi:
                    return -np.inf
                total -= .5*(delta/c.heading_sigma_rad)**2
            total -= .5*((cam[i, 3] - np.pi/2)/c.elevation_sigma_rad)**2
            total -= .5*(cam[i, 5]/c.roll_sigma_rad)**2
            total -= .5*(np.log(cam[i, 6]/self.focal[i])/self.focal_sigma[i])**2
        total -= self.n_ant_label*2*np.log(sigma/c.antenna_label_sigma_px)
        total -= .5*(extra/c.antenna_extra_prior_px)**2
        total -= .5*np.sum((bias/c.gps_common_sigma_m)**2)
        return total

    # --------------------------------------------------------------- reporting

    def height_above_ground(self, ant):
        ant = np.atleast_2d(ant)
        return ant[:, 2] - self.terrain.height(ant[:, 0], ant[:, 1])


# ---------------------------------------------------------------- proposals

# Camera proposal coordinates are (e, n, u, theta, phi, tilt, log f).
# Scales define dimensionless curvature and cap weak-direction proposal SDs;
# they are NOT priors. Metropolis acceptance still uses the exact density.
CAMERA_SCALE = np.array([1., 1., 1., .01, .01, .01, .01])
CAMERA_STEP = np.array([.05, .05, .05, 5e-5, 5e-5, 5e-5, 2e-4])
CAMERA_CURVATURE_FLOOR = 1.


def camera_coordinates(cam):
    """Convert a physical pose (focal pixels) to proposal coordinates."""
    x = np.asarray(cam, float).copy()
    x[6] = np.log(x[6])
    return x


def camera_pose(x):
    """Convert proposal coordinates to a physical pose for projections."""
    cam = np.asarray(x, float).copy()
    with np.errstate(over='ignore', under='ignore'):
        cam[6] = np.exp(cam[6])
    return cam


def _scaled_camera_factor(jacobian, scale=CAMERA_SCALE):
    """Gauss-Newton factor regularized in dimensionless coordinates.

    A floor of one bounds proposal SD by the declared coordinate scales in
    weak directions. Unlike replacing eigenvalues by a fraction of the largest
    raw eigenvalue, this rule does not mix metres, radians and focal pixels.
    """
    scaled = jacobian*scale
    h = scaled.T @ scaled
    w, v = np.linalg.eigh(.5*(h + h.T))
    regularized = np.maximum(w, CAMERA_CURVATURE_FLOOR)
    factor = scale[:, None]*(v/np.sqrt(regularized))
    return factor, dict(eigen_min=float(w.min()), eigen_max=float(w.max()),
                        clipped=int(np.sum(w < CAMERA_CURVATURE_FLOOR)))


def _negative_hessian(logp, x, step):
    """Central-difference negative Hessian of a log density."""
    n = len(x)
    h = np.zeros((n, n))
    base = logp(x)
    if not np.isfinite(base):
        raise ValueError('curvature requested outside the support')
    for a in range(n):
        up, down = x.copy(), x.copy()
        up[a] += step[a]
        down[a] -= step[a]
        h[a, a] = -(logp(up) - 2*base + logp(down))/step[a]**2
    for a in range(n):
        for b in range(a):
            pp, pm, mp, mm = x.copy(), x.copy(), x.copy(), x.copy()
            pp[a] += step[a]; pp[b] += step[b]
            pm[a] += step[a]; pm[b] -= step[b]
            mp[a] -= step[a]; mp[b] += step[b]
            mm[a] -= step[a]; mm[b] -= step[b]
            value = -(logp(pp) - logp(pm) - logp(mp) + logp(mm))/(4*step[a]*step[b])
            h[a, b] = h[b, a] = value
    return h


def _cholesky_from_curvature(h, floor=1e-10):
    """Proposal factor from a curvature estimate, regularized to be usable.

    Regularization is numerical only: a symmetric proposal's scale never enters
    the Metropolis ratio, so a poor factor costs efficiency, never correctness.
    """
    h = .5*(h + h.T)
    w, v = np.linalg.eigh(h)
    w = np.where(w > floor, w, np.maximum(np.max(w), 1.)*1e-6)
    return v*(1/np.sqrt(w))


def camera_geometry(model, cam, ant, points, bias, extra, tx=None, tx_extra=0.,
                    diagnostics=None, step=CAMERA_STEP):
    """Fixed log-f camera proposal factors from residual Jacobians.

    Gauss-Newton curvature avoids indefinite second derivatives of the robust,
    sampled-skyline likelihood. This is the same proposal construction already
    used for landmarks. Factors may be refreshed only during warmup.
    """
    factors = []
    for i in range(model.nc):
        def residual(x):
            pose = camera_pose(x)
            r = model.camera_residuals(i, pose, ant, points, bias, extra,
                                       tx=tx, tx_extra=tx_extra or 0.)
            if r is None:
                return None
            r = np.r_[r, model.horizon_residuals(i, pose)]
            return r if np.all(np.isfinite(r)) else None

        try:
            x = camera_coordinates(cam[i])
            base = residual(x)
            if base is None:
                raise ValueError('curvature requested outside support')
            jacobian = np.empty((len(base), 7))
            one_sided = 0
            for j in range(7):
                up, down = x.copy(), x.copy()
                up[j] += step[j]
                down[j] -= step[j]
                rp, rm = residual(up), residual(down)
                if rp is not None and rm is not None:
                    jacobian[:, j] = (rp-rm)/(2*step[j])
                elif rp is not None:
                    jacobian[:, j] = (rp-base)/step[j]
                    one_sided += 1
                elif rm is not None:
                    jacobian[:, j] = (base-rm)/step[j]
                    one_sided += 1
                else:
                    raise ValueError('both finite-difference poses outside support')
            factor, info = _scaled_camera_factor(jacobian)
            info.update(fallback=False, one_sided=one_sided)
        except (ValueError, np.linalg.LinAlgError) as exc:
            factor = np.diag(.1*CAMERA_SCALE)
            info = dict(fallback=True, reason=str(exc), clipped=None)
        factors.append(factor)
        if diagnostics is not None:
            diagnostics.append(dict(camera=model.keys[i], **info))
    return np.asarray(factors)


def antenna_geometry(model, cam, ant, extra):
    def logp(x):
        return model.antenna_logp(cam, x, extra)
    try:
        return _cholesky_from_curvature(_negative_hessian(logp, ant, np.full(3, .05)))
    except (ValueError, np.linalg.LinAlgError):
        return np.eye(3)*.1


def transmitter_geometry(model, cam, tx, tx_extra=0.):
    def logp(x):
        return model.transmitter_logp(cam, x, tx_extra or 0.)
    try:
        return _cholesky_from_curvature(_negative_hessian(logp, tx, np.full(3, .05)))
    except (ValueError, np.linalg.LinAlgError):
        return np.eye(3)*.1


def landmark_geometry(model, cam, points, step=.05, central=False):
    """Per-landmark 3x3 conditional proposal factors from Gauss-Newton curvature."""
    c = model.config

    def parts(pts):
        pred = np.empty_like(model.xy)
        depth = np.empty(len(model.xy))
        for i, idx in enumerate(model.by_camera):
            pred[idx], depth[idx] = project(cam[i], model.shapes[i], pts[model.op[idx]],
                                            model.distortion[i])
        tie = student_residual((pred - model.xy)/c.tie_sigma_px, c.student_df, 2)
        height = model.terrain.height(pts[:, 0], pts[:, 1])
        terrain = student_residual((pts[:, 2] - height)/c.terrain_sigma_m, c.student_df)
        return tie, terrain

    tie0, terr0 = parts(points)
    d_tie, d_terr = [], []
    for axis in range(3):
        bumped = points.copy()
        bumped[:, axis] += step
        tie1, terr1 = parts(bumped)
        if central:
            bumped[:, axis] -= 2*step
            tie0_axis, terr0_axis = parts(bumped)
            d_tie.append((tie1-tie0_axis)/(2*step))
            d_terr.append((terr1-terr0_axis)/(2*step))
        else:
            d_tie.append((tie1 - tie0)/step)
            d_terr.append((terr1 - terr0)/step)
    jtj = np.zeros((model.npoint, 3, 3))
    for a in range(3):
        for b in range(3):
            jtj[:, a, b] = (np.bincount(model.op, weights=np.sum(d_tie[a]*d_tie[b], axis=1),
                                        minlength=model.npoint) + d_terr[a]*d_terr[b])
    factors = np.empty_like(jtj)
    for j in range(model.npoint):
        factors[j] = _cholesky_from_curvature(jtj[j])
    return factors


def validated_joint_selector(selector, n_state, n_directions):
    """Canonicalize a positive, state-dependent mixture over fixed directions."""
    if selector is None:
        return None
    if set(selector) != {'indices','centers','groups','bandwidth','neutral_mass','bank_floor'}:
        raise ValueError('invalid joint selector fields')
    indices=np.asarray(selector['indices'],int)
    centers=np.asarray(selector['centers'],float)
    groups=np.asarray(selector['groups'],int)
    bandwidth=float(selector['bandwidth'])
    neutral_mass=float(selector['neutral_mass'])
    bank_floor=float(selector['bank_floor'])
    if (indices.ndim!=1 or not len(indices) or len(set(indices.tolist()))!=len(indices)
            or np.any(indices<0) or np.any(indices>=n_state)
            or centers.shape!=(2,len(indices)) or not np.isfinite(centers).all()
            or groups.shape!=(n_directions,) or set(groups.tolist())!={0,1,2}
            or not np.isfinite(bandwidth) or bandwidth<=0
            or not 0<neutral_mass<1 or not 0<bank_floor<.5):
        raise ValueError('invalid joint selector geometry or probabilities')
    return dict(indices=indices.tolist(),centers=centers.tolist(),groups=groups.tolist(),
                bandwidth=bandwidth,neutral_mass=neutral_mass,bank_floor=bank_floor)


def joint_selection_probabilities(state, selector):
    """Select the nearer archived angle bank, with positive reverse mass."""
    state=np.asarray(state,float)
    centers=np.asarray(selector['centers'])
    x=state[np.asarray(selector['indices'],int)]
    squared=np.sum((centers-x)**2,axis=1)
    log_odds=np.clip((squared[1]-squared[0])/(2*selector['bandwidth']**2),-50.,50.)
    near_0=1/(1+np.exp(-log_odds))
    near_0=selector['bank_floor']+(1-2*selector['bank_floor'])*near_0
    masses=np.array([selector['neutral_mass'],
                     (1-selector['neutral_mass'])*near_0,
                     (1-selector['neutral_mass'])*(1-near_0)])
    groups=np.asarray(selector['groups'],int)
    counts=np.bincount(groups,minlength=3)
    probabilities=masses[groups]/counts[groups]
    if not np.isfinite(probabilities).all() or np.any(probabilities<=0):
        raise RuntimeError('invalid joint selection probabilities')
    return probabilities


class Chain:
    """Blocked Metropolis-within-Gibbs state for one chain."""

    def __init__(self, model, start, rng, difference_step_factor=1.,
                 joint_directions=None, joint_scale=.005, joint_selector=None):
        self.model = model
        self.rng = rng
        if not np.isfinite(model.logp(start)):
            raise ValueError('chain start is outside target support')
        if not 0 < difference_step_factor <= 1:
            raise ValueError('difference step factor must be in (0, 1]')
        self.difference_step_factor = difference_step_factor
        self.joint_directions = (np.empty((0,len(start))) if joint_directions is None
                                 else np.asarray(joint_directions,float).copy())
        if (self.joint_directions.ndim != 2 or self.joint_directions.shape[1] != len(start)
                or not np.isfinite(self.joint_directions).all() or joint_scale <= 0):
            raise ValueError('invalid frozen joint directions or scale')
        self.joint_scale = np.full(len(self.joint_directions),joint_scale)
        self.joint_selector = validated_joint_selector(joint_selector,len(start),len(self.joint_directions))
        # Separate stream preserves the block stream when the joint kernel is enabled.
        seed = int.from_bytes(hashlib.sha256(pickle.dumps(rng.bit_generator.state)).digest()[:8],'little')
        self.joint_rng = np.random.default_rng(seed)
        self.joint_counts = {k:np.zeros(len(self.joint_directions)) for k in
                             ['proposals','accepts','support_rejections','proposed_sq','accepted_sq']}

        (self.cam, self.ant, self.tx, self.bias, self.extra,
         self.tx_extra, self.points) = model.unpack(start)
        if self.tx is not None:
            self.tx = self.tx.copy()
        self.cam = self.cam.copy()
        self.points = self.points.copy()
        self.horizon = np.array([model.horizon_logp(i, self.cam[i]) for i in range(model.nc)])
        self.camera_geometry_diagnostics = []
        self.camera_factor = camera_geometry(model, self.cam, self.ant, self.points,
                                             self.bias, self.extra, tx=self.tx,
                                             tx_extra=self.tx_extra,
                                             diagnostics=self.camera_geometry_diagnostics,
                                             step=CAMERA_STEP*self.difference_step_factor)
        self.antenna_factor = antenna_geometry(model, self.cam, self.ant, self.extra)
        self.transmitter_factor = (transmitter_geometry(model, self.cam, self.tx, self.tx_extra)
                                   if self.tx is not None else None)
        self.point_factor = landmark_geometry(model, self.cam, self.points,
                                               step=.05*self.difference_step_factor,
                                               central=self.difference_step_factor != 1.)
        self.camera_scale = np.full(model.nc, 2.38/np.sqrt(7))
        self.antenna_scale = 2.38/np.sqrt(3)
        self.transmitter_scale = 2.38/np.sqrt(3)
        self.point_scale = 2.38/np.sqrt(3)
        self.extra_scale = 1.
        self.tx_extra_scale = 1.
        self.shift_scale = .5
        self.counts = dict(camera=np.zeros(model.nc), camera_n=np.zeros(model.nc),
                           camera_proposed_logf_sq=np.zeros(model.nc),
                           camera_jump_logf_sq=np.zeros(model.nc),
                           camera_support_reject=np.zeros(model.nc),
                           antenna=0., antenna_n=0., transmitter=0., transmitter_n=0.,
                           tx_extra=0., tx_extra_n=0.,
                           point=0., point_n=0.,
                           extra=0., extra_n=0., shift=0., shift_n=0.)

    # ------------------------------------------------------------- blocks

    def update_cameras(self):
        m, rng = self.model, self.rng
        for i in rng.permutation(m.nc):
            current = m.camera_logp(i, self.cam[i], self.ant, self.points, self.bias,
                                    self.extra, horizon=self.horizon[i], tx=self.tx,
                                    tx_extra=self.tx_extra or 0.)
            x = camera_coordinates(self.cam[i])
            delta = self.camera_scale[i]*(self.camera_factor[i] @ rng.standard_normal(7))
            proposal = camera_pose(x + delta)
            self.counts['camera_proposed_logf_sq'][i] += delta[6]**2
            if not m._camera_support(proposal):
                self.counts['camera_n'][i] += 1
                self.counts['camera_support_reject'][i] += 1
                continue
            horizon = m.horizon_logp(i, proposal)
            candidate = m.camera_logp(i, proposal, self.ant, self.points, self.bias,
                                      self.extra, horizon=horizon, tx=self.tx,
                                      tx_extra=self.tx_extra or 0.)
            self.counts['camera_n'][i] += 1
            # Symmetric in log f: the target already has a Gaussian log-f prior.
            # No Jacobian belongs here. A raw-f symmetric proposal would instead
            # require -log(f_new/f_old) to sample this same intended target.
            if np.log(rng.random()) < candidate - current:
                self.cam[i] = proposal
                self.horizon[i] = horizon
                self.counts['camera'][i] += 1
                self.counts['camera_jump_logf_sq'][i] += delta[6]**2

    def update_antenna(self, repeats=8):
        m, rng = self.model, self.rng
        current = m.antenna_logp(self.cam, self.ant, self.extra)
        for _ in range(repeats):
            proposal = self.ant + self.antenna_scale*(self.antenna_factor @ rng.standard_normal(3))
            candidate = m.antenna_logp(self.cam, proposal, self.extra)
            self.counts['antenna_n'] += 1
            if np.log(rng.random()) < candidate - current:
                self.ant, current = proposal, candidate
                self.counts['antenna'] += 1

    def update_transmitter(self, repeats=8):
        """Transmitter block. Mirrors the antenna block: the transmitter appears
        in no horizon and no feature row, so a move costs only the projections
        into the eight views that carry a transmitter pick."""
        m, rng = self.model, self.rng
        if self.tx is None:
            return
        current = m.transmitter_logp(self.cam, self.tx, self.tx_extra)
        for _ in range(repeats):
            proposal = self.tx + self.transmitter_scale*(self.transmitter_factor @ rng.standard_normal(3))
            candidate = m.transmitter_logp(self.cam, proposal, self.tx_extra)
            self.counts['transmitter_n'] += 1
            if np.log(rng.random()) < candidate - current:
                self.tx, current = proposal, candidate
                self.counts['transmitter'] += 1

    def update_tx_extra(self, repeats=4):
        """Excess scatter of the transmitter picks, mirroring update_extra."""
        m, rng = self.model, self.rng
        if self.tx_extra is None:
            return
        current = m.tx_scatter_logp(self.cam, self.tx, self.tx_extra)
        for _ in range(repeats):
            proposal = self.tx_extra + self.tx_extra_scale*rng.standard_normal()
            candidate = m.tx_scatter_logp(self.cam, self.tx, proposal)
            self.counts['tx_extra_n'] += 1
            if np.log(rng.random()) < candidate - current:
                self.tx_extra, current = proposal, candidate
                self.counts['tx_extra'] += 1

    def update_bias(self):
        """Exact Gibbs draw: the common GPS bias is Gaussian given everything else."""
        m, rng = self.model, self.rng
        use = m.has_gps
        if not use.any():
            return
        weight = 1/m.gps_sigma[use]**2
        precision = weight.sum() + 1/m.config.gps_common_sigma_m**2
        mean = ((m.gps[use] - self.cam[use, :2])*weight[:, None]).sum(axis=0)/precision
        self.bias = mean + rng.standard_normal(2)/np.sqrt(precision)

    def update_extra(self, repeats=4):
        m, rng = self.model, self.rng
        current = m.scatter_logp(self.cam, self.ant, self.extra)
        for _ in range(repeats):
            proposal = self.extra + self.extra_scale*rng.standard_normal()
            candidate = m.scatter_logp(self.cam, self.ant, proposal)
            self.counts['extra_n'] += 1
            if np.log(rng.random()) < candidate - current:
                self.extra, current = proposal, candidate
                self.counts['extra'] += 1

    def update_points(self):
        m, rng = self.model, self.rng
        current = m.point_logp(cam=self.cam, points=self.points)
        step = np.einsum('jab,jb->ja', self.point_factor, rng.standard_normal((m.npoint, 3)))
        proposal = self.points + self.point_scale*step
        candidate = m.point_logp(cam=self.cam, points=proposal)
        take = np.log(rng.random(m.npoint)) < candidate - current
        self.points[take] = proposal[take]
        self.counts['point'] += take.sum()
        self.counts['point_n'] += m.npoint

    def update_shift(self):
        """Translate the whole network: cameras, antenna and landmarks together.

        This is the move that explores the datum direction.  Only terms tied to
        the DEM and to EXIF -- horizons, GPS, landmark heights -- resist it, so
        it is the natural coordinate for the weak horizontal direction found in
        the view geometry.  It is the one move that needs a full density
        evaluation, so it runs on a schedule rather than every sweep.
        """
        m, rng = self.model, self.rng
        current = m.logp(m.pack(self.cam, self.ant, self.tx, self.bias, self.extra,
                                self.tx_extra, self.points))
        delta = self.shift_scale*rng.standard_normal(3)
        cam = self.cam.copy()
        cam[:, :3] += delta
        ant = self.ant + delta
        tx = None if self.tx is None else self.tx + delta
        points = self.points + delta
        candidate = m.logp(m.pack(cam, ant, tx, self.bias, self.extra, self.tx_extra, points))
        self.counts['shift_n'] += 1
        if np.log(rng.random()) < candidate - current:
            self.cam, self.ant, self.points = cam, ant, points
            self.tx = tx
            self.horizon = np.array([m.horizon_logp(i, cam[i]) for i in range(m.nc)])
            self.counts['shift'] += 1

    def update_joint(self):
        """Fixed-direction random walk in the stored log-f measure.

        State-dependent direction weights include their reverse/forward ratio
        in the exact Metropolis-Hastings test. Only scalar scales adapt during
        discarded warmup. Refresh cached horizons after acceptance.
        """
        if not len(self.joint_directions):
            return
        rng,m=self.joint_rng,self.model
        selector=getattr(self,'joint_selector',None)
        current=self.state()
        probabilities=(joint_selection_probabilities(current,selector)
                       if selector is not None else None)
        j=(int(rng.choice(len(self.joint_directions),p=probabilities))
           if probabilities is not None else int(rng.integers(len(self.joint_directions))))
        amplitude=float(self.joint_scale[j]*rng.standard_normal())
        proposal=current+amplitude*self.joint_directions[j]
        old=float(m.logp(current));new=float(m.logp(proposal))
        if not np.isfinite(old):
            raise RuntimeError('joint kernel reached an invalid current state')
        self.joint_counts['proposals'][j]+=1
        self.joint_counts['proposed_sq'][j]+=amplitude**2
        if not np.isfinite(new):
            self.joint_counts['support_rejections'][j]+=1
        logu=np.log(rng.random())
        log_selection_ratio=(np.log(joint_selection_probabilities(proposal,selector)[j])
                             -np.log(probabilities[j]) if selector is not None and np.isfinite(new) else 0.)
        if np.isfinite(new) and logu < new-old+log_selection_ratio:
            (self.cam,self.ant,self.tx,self.bias,self.extra,self.tx_extra,self.points)=m.unpack(proposal)
            self.horizon=np.array([m.horizon_logp(i,self.cam[i]) for i in range(m.nc)])
            self.joint_counts['accepts'][j]+=1
            self.joint_counts['accepted_sq'][j]+=amplitude**2

    def joint_acceptance(self):
        n=np.maximum(self.joint_counts['proposals'],1)
        return dict(**{k:v.tolist() for k,v in self.joint_counts.items()},
                    acceptance=(self.joint_counts['accepts']/n).tolist(),
                    accepted_rms=np.sqrt(self.joint_counts['accepted_sq']/n).tolist(),
                    scale=self.joint_scale.tolist())

    # -------------------------------------------------------------- driver

    def sweep(self, shift=False, joint=False):
        self.update_cameras()
        self.update_points()
        self.update_antenna()
        self.update_transmitter()
        self.update_tx_extra()
        self.update_bias()
        self.update_extra()
        if shift:
            self.update_shift()
        if joint:
            self.update_joint()

    def adapt(self, rate):
        def tune(scale, accepted, total, target):
            # The camera block keeps one count per camera, so `total` is an
            # array there and a scalar everywhere else; a bare `total < 1`
            # raises on the array. Clamp for the division and leave the scale
            # untouched wherever nothing was proposed.
            total = np.asarray(total, float)
            accepted = np.asarray(accepted, float)
            factor = np.where(total < 1, 1.,
                              np.exp(rate*(accepted/np.maximum(total, 1.) - target)))
            return scale*(factor if factor.ndim else float(factor))
        self.camera_scale = tune(self.camera_scale, self.counts['camera'],
                                 np.maximum(self.counts['camera_n'], 1), .234)
        self.antenna_scale = tune(self.antenna_scale, self.counts['antenna'],
                                  max(self.counts['antenna_n'], 1), .3)
        if self.tx is not None:
            self.transmitter_scale = tune(self.transmitter_scale, self.counts['transmitter'],
                                          max(self.counts['transmitter_n'], 1), .3)
            self.tx_extra_scale = tune(self.tx_extra_scale, self.counts['tx_extra'],
                                       max(self.counts['tx_extra_n'], 1), .44)
        self.point_scale = tune(self.point_scale, self.counts['point'],
                                max(self.counts['point_n'], 1), .35)
        self.extra_scale = tune(self.extra_scale, self.counts['extra'],
                                max(self.counts['extra_n'], 1), .44)
        self.shift_scale = tune(self.shift_scale, self.counts['shift'],
                                max(self.counts['shift_n'], 1), .3)
        self.joint_scale = tune(self.joint_scale, self.joint_counts['accepts'],
                                self.joint_counts['proposals'], .44)
        for key in self.joint_counts:
            self.joint_counts[key].fill(0.)
        for key in self.counts:
            self.counts[key] = np.zeros(self.model.nc) if key.startswith('camera') else 0.

    def acceptance(self):
        n = np.maximum(self.counts['camera_n'], 1)
        return dict(camera=float(np.sum(self.counts['camera'])/max(np.sum(self.counts['camera_n']), 1)),
                    camera_per_camera=(self.counts['camera']/n).tolist(),
                    camera_proposals=self.counts['camera_n'].astype(int).tolist(),
                    camera_accepts=self.counts['camera'].astype(int).tolist(),
                    camera_proposed_logf_rms=np.sqrt(self.counts['camera_proposed_logf_sq']/n).tolist(),
                    camera_jump_logf_rms=np.sqrt(self.counts['camera_jump_logf_sq']/n).tolist(),
                    camera_support_reject_per_camera=(self.counts['camera_support_reject']/n).tolist(),
                    antenna=float(self.counts['antenna']/max(self.counts['antenna_n'], 1)),
                    transmitter=float(self.counts['transmitter']/max(self.counts['transmitter_n'], 1)),
                    tx_extra=float(self.counts['tx_extra']/max(self.counts['tx_extra_n'], 1)),
                    point=float(self.counts['point']/max(self.counts['point_n'], 1)),
                    extra=float(self.counts['extra']/max(self.counts['extra_n'], 1)),
                    shift=float(self.counts['shift']/max(self.counts['shift_n'], 1)))

    def state(self):
        return self.model.pack(self.cam, self.ant, self.tx, self.bias, self.extra,
                               self.tx_extra, self.points)


# ------------------------------------------------------------ initialization

def horizontal_axes(model, antenna=None):
    """Weak and strong horizontal directions of the labelled-view geometry.

    Each view constrains the antenna across its own line of sight but not along
    it.  Summing the across-sight projectors gives a cross-range information
    matrix whose smallest eigenvector is the direction the view set constrains
    least.  This is descriptive geometry, not a fit, and it is used only to
    choose overdispersed starting points along the direction where chain
    disagreement is most likely.
    """
    antenna = model.antenna0 if antenna is None else antenna
    # Only the views carrying an antenna label constrain the antenna.
    labelled = model.cameras0[model.has_ant_label] if hasattr(model, 'has_ant_label') else model.cameras0
    offsets = antenna[None, :2] - labelled[:, :2]
    unit = offsets/np.linalg.norm(offsets, axis=1, keepdims=True)
    information = np.sum(np.eye(2)[None] - unit[:, :, None]*unit[:, None, :], axis=0)
    values, vectors = np.linalg.eigh(information)
    return vectors[:, 0], vectors[:, 1], values


def dispersed_start(model, index):
    """Overdispersed antenna starts, per Gelman-Rubin.

    Chains 0 and 1 start at the bundle-adjustment solution.  The rest displace
    the antenna alone by several metres along the weak horizontal direction, the
    strong horizontal direction, and in altitude.  The displacements bracket the
    3.42 m separation between the bundle-adjustment and 13-camera pilot antennas,
    so chains that agree afterwards have crossed that gap rather than assumed it
    away.  Cameras and landmarks start identical in every chain, which isolates
    the antenna's own mixing.
    """
    start = model.start_vector()
    weak, strong, _ = horizontal_axes(model)
    offsets = {0: np.zeros(3), 1: np.zeros(3),
               2: np.r_[+6.*weak, 0.], 3: np.r_[-6.*weak, 0.],
               4: np.r_[+6.*strong, 0.], 5: np.r_[-6.*strong, 0.],
               6: np.array([0., 0., +3.]), 7: np.array([0., 0., -3.])}
    start[7*model.nc:7*model.nc+3] += offsets[index % 8]
    return start


# ------------------------------------------------------------------- driver

def manifest(model, args):
    files = {}
    for path in model.input_files:
        path = Path(path)
        if path.exists():
            files[str(path)] = digest(path)
    return dict(product=('marjum_joint_antenna_transmitter_posterior' if model.joint
                         else 'marjum_antenna_posterior_91m'),
                campaign='marjum-2026-07',
                joint=model.joint,
                geometry_release=release_check(model.state_file),
                antenna_labelled_views_n=model.n_ant_label,
                transmitter_labelled_views_n=model.n_tx_label,
                antenna_labelled_keys=[k for k, ok in zip(model.keys, model.has_ant_label) if ok],
                transmitter_labelled_keys=[k for k, ok in zip(model.keys, model.has_tx_label) if ok],
                state_file=str(model.state_file),
                feature_dir=str(model.feature_dir),
                dem_file=model.dem_file,
                keys=model.keys,
                cameras=model.nc,
                landmarks=model.npoint,
                observations=int(len(model.xy)),
                cameras_without_gps_or_heading_prior=model.unpriored,
                distortion='fixed per lens group from the bundle-adjustment product',
                camera_proposal=dict(version='logf_scaled_gauss_newton_v1',
                                     coordinates=['e_m', 'n_m', 'u_m', 'theta_rad',
                                                  'phi_rad', 'tilt_rad', 'log_f'],
                                     scale=CAMERA_SCALE.tolist(),
                                     finite_difference_step=CAMERA_STEP.tolist(),
                                     dimensionless_curvature_floor=CAMERA_CURVATURE_FLOOR,
                                     target_measure='Gaussian prior in log focal length'),
                config=asdict(model.config),
                input_sha256=files,
                args=vars(args))


def run_chain(model, index, tune, draws, seed, out, thin_points=100, shift_every=5,
              refresh=(0.3, 0.6), checkpoint_every=100, resume=False, start=None,
              difference_step_factor=1., joint_directions=None, joint_scale=.005,
              joint_every=1, joint_selector=None):
    checkpoint = Path(out)/f'checkpoint_{index}.pkl'
    initial = dispersed_start(model,index) if start is None else np.asarray(start,float).copy()
    array_hash = lambda a: None if a is None else hashlib.sha256(np.asarray(a,dtype=float).tobytes()).hexdigest()
    if joint_every < 1:
        raise ValueError('joint_every must be positive')
    joint_selector=validated_joint_selector(joint_selector,len(initial),
        0 if joint_directions is None else len(joint_directions))
    signature = dict(start_sha256=array_hash(initial),joint_directions_sha256=array_hash(joint_directions),
                     difference_step_factor=difference_step_factor,joint_scale=joint_scale,joint_every=joint_every,
                     index=index, tune=tune, draws=draws, seed=seed,
                     thin_points=thin_points, shift_every=shift_every, refresh=refresh,
                     config=asdict(model.config),
                     inputs={str(p): digest(p) for p in model.input_files if Path(p).exists()})
    if joint_selector is not None:
        signature['joint_selector']=joint_selector
    if checkpoint.exists() and not resume:
        raise FileExistsError(f'{checkpoint} exists; use resume=True or a fresh directory')
    rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
    refresh_at = {int(tune*f) for f in refresh}
    if resume:
        # Read only this run's trusted local checkpoint. Pickle is necessary for
        # exact Generator/array restoration; never load a downloaded checkpoint.
        with checkpoint.open('rb') as f:
            saved = pickle.load(f)
        if saved['signature'] != signature:
            raise ValueError('checkpoint code, inputs or run configuration changed')
        chain = Chain.__new__(Chain)
        chain.__dict__.update(saved['chain'])
        chain.model = model
        records, stats, landmarks = saved['records'], saved['stats'], saved['landmarks']
        geometry_history = saved['geometry_history']
        first_iteration = saved['next_iteration']
    else:
        chain = Chain(model, initial, rng, difference_step_factor=difference_step_factor,
                      joint_directions=joint_directions,joint_scale=joint_scale,
                      joint_selector=joint_selector)
        records, stats, landmarks = [], [], []
        geometry_history = [dict(iteration=-1, cameras=chain.camera_geometry_diagnostics)]
        first_iteration = 0
    clock = time.monotonic()
    for iteration in range(first_iteration, tune + draws):
        if iteration == tune:
            # Reset BEFORE the first retained sweep so all retained proposals
            # contribute to the reported acceptance and jump-size diagnostics.
            chain.counts = {k: (np.zeros(model.nc) if k.startswith('camera') else 0.)
                            for k in chain.counts}
            for key in chain.joint_counts:
                chain.joint_counts[key].fill(0.)
        chain.sweep(shift=(iteration % shift_every == 0),joint=(iteration % joint_every == 0))
        if iteration < tune:
            if (iteration + 1) % 25 == 0:
                # Vanishing adaptation during discarded warmup only; the kernel
                # is frozen for every retained draw.
                chain.adapt(min(2., 5/np.sqrt((iteration + 1)/25)))
            if iteration in refresh_at:
                chain.camera_geometry_diagnostics = []
                chain.camera_factor = camera_geometry(model, chain.cam, chain.ant,
                                                      chain.points, chain.bias, chain.extra,
                                                      tx=chain.tx, tx_extra=chain.tx_extra,
                                                      diagnostics=chain.camera_geometry_diagnostics,
                                                      step=CAMERA_STEP*chain.difference_step_factor)
                geometry_history.append(dict(iteration=iteration,
                                             cameras=chain.camera_geometry_diagnostics))
                chain.antenna_factor = antenna_geometry(model, chain.cam, chain.ant, chain.extra)
                if chain.tx is not None:
                    chain.transmitter_factor = transmitter_geometry(model, chain.cam, chain.tx,
                                                                    chain.tx_extra)
                chain.point_factor = landmark_geometry(model, chain.cam, chain.points,
                    step=.05*chain.difference_step_factor,central=chain.difference_step_factor != 1.)
        else:
            records.append(chain.state()[:model.ng].copy())
            if (iteration - tune) % thin_points == 0:
                # A full joint density costs a whole skyline pass, so the
                # log-posterior trace is thinned; it is a convergence
                # diagnostic, not part of the reported posterior.
                stats.append((iteration - tune, model.logp(chain.state())))
                landmarks.append(chain.points.copy())
        if (iteration + 1) % 100 == 0:
            rate = chain.acceptance()
            print(f'chain {index} {iteration+1}/{tune+draws} '
                  f'ant={np.round(chain.ant, 3)} extra={chain.extra:.1f} '
                  f'acc cam={rate["camera"]:.2f} ant={rate["antenna"]:.2f} '
                  f'pt={rate["point"]:.2f} shift={rate["shift"]:.2f} '
                  f'{time.monotonic()-clock:.0f}s', flush=True)
        if checkpoint_every and ((iteration + 1) % checkpoint_every == 0 or
                                 iteration + 1 == tune + draws):
            # No model/DEM duplication. Save after all adaptation and recording,
            # so restart resumes at the next sweep with identical RNG state.
            saved = dict(signature=signature, next_iteration=iteration+1,
                         chain={k: v for k, v in vars(chain).items() if k != 'model'},
                         records=records, stats=stats, landmarks=landmarks,
                         geometry_history=geometry_history)
            temporary = checkpoint.with_suffix('.pkl.tmp')
            with temporary.open('wb') as f:
                pickle.dump(saved, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush()
                os.fsync(f.fileno())
            temporary.replace(checkpoint)
    values = np.array(records)
    np.savez_compressed(Path(out)/f'chain_{index}.npz',
                        globals=values, logp=np.array(stats),
                        landmarks=np.array(landmarks),
                        start=initial[:model.ng],
                        joint_acceptance=json.dumps(chain.joint_acceptance()),
                        difference_step_factor=np.array(difference_step_factor),
                        acceptance=json.dumps(chain.acceptance()),
                        camera_keys=np.asarray(model.keys),
                        camera_factor=chain.camera_factor,
                        camera_scale=chain.camera_scale,
                        camera_geometry_history=json.dumps(geometry_history),
                        final=chain.state())
    return values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--chain', type=int)
    parser.add_argument('--tune', type=int, default=3000)
    parser.add_argument('--draws', type=int, default=7000)
    parser.add_argument('--seed', type=int, default=20260913)
    parser.add_argument('--skyline-samples', type=int, default=Config.skyline_samples)
    parser.add_argument('--antenna-extra-prior-px', type=float,
                        default=Config.antenna_extra_prior_px)
    parser.add_argument('--shift-every', type=int, default=5)
    parser.add_argument('--checkpoint-every', type=int, default=100,
                        help='Atomically save restart state every N sweeps (0 disables)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume this run from its trusted local checkpoint')
    parser.add_argument('--state-file', default=STATE_FILE)
    parser.add_argument('--dem-file', default=DEM_FILE,
                        help='DEM cache used by this state (default: legacy southwest mosaic)')
    parser.add_argument('--feature-dir', default='cv_features',
                        help='Feature cache containing the horizon observations for this state')
    parser.add_argument('--meta-file', default='meta.json',
                        help='Image labels (ant_px, transmitter_px) for this state')
    parser.add_argument('--exif-file', default='marjum_2026_07_exif.npz',
                        help='EXIF prior cache; use marjum_2026_07_exif_joint.npz '
                             'for a 29-pose joint state, which needs priors for the '
                             'six transmitter-era cameras too')
    args = parser.parse_args(argv)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    config = Config(skyline_samples=args.skyline_samples,
                    antenna_extra_prior_px=args.antenna_extra_prior_px)
    model = Posterior(state_file=args.state_file, dem_file=args.dem_file, meta_file=args.meta_file,
                      exif_file=args.exif_file, config=config, feature_dir=args.feature_dir)

    target = out/'manifest.json'
    if not target.exists():
        target.write_text(json.dumps(manifest(model, args), indent=2, default=str) + '\n')
    if args.chain is None:
        parser.error('give --chain N; run chains as separate processes and combine after')
    result = out/f'chain_{args.chain}.npz'
    if result.exists():
        raise FileExistsError(f'{result} exists; use a fresh output directory')
    run_chain(model, args.chain, args.tune, args.draws, args.seed, out,
              shift_every=args.shift_every, checkpoint_every=args.checkpoint_every,
              resume=args.resume)


if __name__ == '__main__':
    main()
