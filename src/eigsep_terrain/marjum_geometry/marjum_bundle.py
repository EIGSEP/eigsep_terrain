"""Parallax-aware, DEM-anchored initialization for the Marjum notebooks.

Pixels are (x=column, y=row), in the existing notebooks' bottom-up images.
This is a deterministic initialization objective, NOT a calibrated posterior.
Only new output files are written. Existing poses, labels and segmentations
are inputs. See Marjum 2026-07 CV Initialization.ipynb for interpretation.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from itertools import combinations
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

PRM_ORDER = ('e', 'n', 'u', 'th', 'ph', 'ti', 'f')


def rotation(p):
    """Body-to-ENU rotation, identical to HorizonImage.get_rays."""
    th, ph, ti = p[3:6]
    def rz(t):
        c, s = np.cos(t), np.sin(t)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
    c, s = np.cos(th), np.sin(th)
    return rz(ph) @ np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]) @ rz(ti)


def rays(p, shape, xy):
    h, w = shape
    xy = np.asarray(xy).reshape(-1, 2)
    v = np.column_stack((h // 2 - xy[:, 1], w // 2 - xy[:, 0],
                         np.full(len(xy), p[6]))) @ rotation(p).T
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def project(p, shape, xyz):
    """Return xy and signed optical depth; negative-depth points are invalid."""
    h, w = shape
    v = (np.atleast_2d(xyz) - p[:3]) @ rotation(p)
    z = v[:, 2]
    # Finite values during optimization, with an explicit cheirality penalty.
    denom = np.maximum(z, 0.1)
    return np.column_stack((w // 2 - p[6] * v[:, 1] / denom,
                            h // 2 - p[6] * v[:, 0] / denom)), z


# The canonical working grid is anchored at marjum_dem.npz's corner: that cache
# reports E from 0 and N from 0. The _south/_sw caches extend that mosaic
# 1000 m south / south+west, so their working-grid corners are the offsets
# below. The offsets were originally registered by exact submatrix alignment
# (main == south[2000:,:] == sw[2000:,2000:]).
WORKING_GRID_SHIFT_M = {
    'marjum_dem.npz': (0., 0.),
    'marjum_dem_south.npz': (0., -1000.),
    'marjum_dem_sw.npz': (-1000., -1000.),
}


def working_grid(dem):
    """Anchor a MarjumDEM cache to the canonical working grid, in place.

    Afterwards get_en() and interp_alt() accept/produce working-grid
    coordinates regardless of which cache backs the DEM, and regardless of
    whether MarjumDEM already anchored the cache itself. Idempotent; DEM
    stand-ins without a cache file are left untouched.
    """
    if getattr(dem, '_working_shift', None) is not None:
        return dem
    cache = getattr(dem, '_cache_file', None)
    if cache is None:
        return dem
    name = Path(cache).name
    if name not in WORKING_GRID_SHIFT_M:
        raise ValueError(f'no registered working-grid anchor for {name}')
    se, sn = WORKING_GRID_SHIFT_M[name]

    # Decide whether the cache is ALREADY anchored from the DEM's *reported
    # geo-bounds*, never from its raw array contents.
    #
    # eigsep_terrain gained its own geo-anchoring for the _sw/_south caches on
    # 2026-09-17 (merge 343e91d). That left the stored arrays byte-identical,
    # so the submatrix-alignment check this function used to rely on could not
    # see the change and kept applying a shift on top of one already applied --
    # double-shifting by up to 1000 m. The failure was silent: the camera
    # landed ~490 m underground, skyline() returned 89.88 deg at every azimuth,
    # and a terrain fit produced a flat, plausible-looking cost surface rather
    # than an error. Reported twice (MEMO-012 section 3) before being fixed.
    #
    # Reported bounds are the right discriminator because they are exactly what
    # changed, and what downstream callers actually consume.
    east, north = dem.get_en()
    e0, n0 = float(east[0]), float(north[0])
    tol = float(dem.res)
    if abs(e0 - se) <= tol and abs(n0 - sn) <= tol:
        # Already on the working grid (this is also the correct no-op path for
        # marjum_dem.npz, whose shift is (0, 0)).
        dem._working_shift = (se, sn)
        return dem
    if abs(e0) > tol or abs(n0) > tol:
        raise ValueError(
            f'{name} reports its corner at E {e0}, N {n0}, which is neither the '
            f'unanchored (0, 0) nor the expected working-grid anchor {(se, sn)}. '
            f'Refusing to guess -- the registered offsets in '
            f'WORKING_GRID_SHIFT_M need review against this eigsep_terrain.')

    if (se, sn) != (0., 0.):
        dr, dc = int(round(-sn / dem.res)), int(round(-se / dem.res))
        ref = np.load(Path(cache).with_name('marjum_dem.npz'))['dem']
        if not np.array_equal(ref[4000:4064, 3200:3264],
                              dem.data[4000 + dr:4064 + dr, 3200 + dc:3264 + dc]):
            raise ValueError(f'{name} does not align with marjum_dem.npz at the registered shift {(se, sn)}')
        del ref
        orig_get_en = dem.get_en
        orig_interp_alt = dem.interp_alt
        def get_en(*args, **kwargs):
            if args or kwargs:
                raise ValueError('working-grid get_en supports no-argument calls only')
            E, N = orig_get_en()
            return E + se, N + sn
        def interp_alt(e_m, n_m, **kwargs):
            return orig_interp_alt(np.asarray(e_m) - se, np.asarray(n_m) - sn, **kwargs)
        dem.get_en = get_en
        dem.interp_alt = interp_alt
    dem._working_shift = (se, sn)
    return dem


class Terrain:
    """Bilinear DEM sampling in canonical working-grid coordinates."""
    def __init__(self, dem):
        self.dem = working_grid(dem)
        self.e, self.n = self.dem.get_en()
        self.res = float(dem.res)
        # map_coordinates otherwise rounds interpolated integer DEMs to integers.
        self.data = np.asarray(dem.data, dtype=np.float32)

    def height(self, e, n):
        e, n = np.broadcast_arrays(e, n)
        rc = np.array([(n.ravel() - self.n[0]) / self.res,
                       (e.ravel() - self.e[0]) / self.res])
        return map_coordinates(self.data, rc, order=1, mode='constant',
                               cval=np.nan, prefilter=False, output=np.float64).reshape(e.shape)

    def skyline(self, cam, azimuth, count=384):
        """Elevation of highest DEM point along each bearing, within the tile.

        Sample every ~1 m nearby and geometrically farther away. The maximizing
        terrain point can change with camera pose; it is never a fixed landmark.
        This finite-resolution skyline is only a surrogate for full ray tracing.
        """
        azimuth = np.atleast_1d(azimuth)
        ce, sn = np.cos(azimuth), np.sin(azimuth)
        def edge_distance(c, lo, hi, direction):
            # Clamp magnitude, not sign: a tiny negative direction must still
            # divide by a tiny negative safe value, or the branch selected by
            # `direction >= 0` disagrees with the sign actually used.
            safe = np.where(np.abs(direction) > 1e-12, direction,
                            np.copysign(1e-12, direction))
            return np.where(direction >= 0, (hi - c) / safe, (lo - c) / safe)
        end = np.minimum(edge_distance(cam[0], self.e[0], self.e[-1], ce),
                         edge_distance(cam[1], self.n[0], self.n[-1], sn))
        end = np.maximum(end - self.res, 2.)
        # Independent of the optimizer's current elevation, with no dropped rows.
        fraction = np.linspace(0., 1., count)
        dist = np.exp(np.log(1.) + np.log(end[:, None]) * fraction)
        z = self.height(cam[0] + ce[:, None] * dist, cam[1] + sn[:, None] * dist)
        angle = np.arctan2(z - cam[2], dist)
        return np.max(np.where(np.isfinite(angle), angle, -np.pi / 2), axis=1)

    def intersect(self, p, shape, xy):
        """Use the library ray tracer only to initialize feature depths."""
        from eigsep_terrain.ray_numba import ray_distance_coarse_to_fine_numba
        d = rays(p, shape, xy)
        distance = ray_distance_coarse_to_fine_numba(
            self.e, self.n, self.dem.data, np.asarray(p[:3], np.float32),
            np.asarray(d.T, np.float32))
        return p[:3] + distance[:, None] * d


def boundary_pixels(sky, tree=None, spacing=150, exclude=None):
    """Sample actual sky/ground transitions; omit all-ground/all-sky columns."""
    h, w = sky.shape
    x = np.arange(spacing // 2, w, spacing)
    y = np.where(~sky[:, x], np.arange(h)[:, None], -1).max(axis=0)
    valid = (y >= 0) & (y < h - 1)
    if tree is not None:
        valid &= tree[np.clip(y, 0, h - 1), x] < 0.15
    if exclude is not None:
        if np.shape(exclude) != sky.shape:
            raise ValueError('horizon exclusion/image shape mismatch')
        valid &= ~np.asarray(exclude, bool)[np.clip(y, 0, h - 1), x]
    return np.column_stack((x[valid], y[valid] + 0.5)).astype(float)


def extract_features(root, keys, cache, max_size=1600, nfeatures=5000):
    """Extract each image once; retain small descriptors, not full RGB arrays."""
    from eigsep_terrain.imageio import load_image
    cache.mkdir(parents=True, exist_ok=True)
    result = {}
    for key in keys:
        photo = root / f'marjum-2026-07/IMG_{key}.HEIC'
        seg = root / f'img_seg_IMG_{key}.npz'
        signature = np.array([photo.stat().st_mtime_ns, seg.stat().st_mtime_ns,
                              max_size, nfeatures, 3], dtype=np.int64)
        path = cache / f'sift_{key}.npz'
        if path.exists():
            with np.load(path) as z:
                if np.array_equal(z['signature'], signature):
                    result[key] = {k: z[k] for k in z.files}
                    continue
        rgb = np.flipud(load_image(str(photo))).copy()
        h, w = rgb.shape[:2]
        with np.load(seg) as z:
            sky = np.flipud(z['skymask']).astype(bool)
            tree = np.flipud(z['ptree'])
            exclude = np.flipud(z['horizon_exclude']).astype(bool) if 'horizon_exclude' in z else np.zeros(sky.shape, bool)
        if sky.shape != (h, w):
            raise ValueError(f'{key}: segmentation/image orientation mismatch')
        horizon = boundary_pixels(sky, tree, exclude=exclude)
        factor = min(1., max_size / max(h, w))
        small = cv2.resize(rgb, (round(w * factor), round(h * factor)),
                           interpolation=cv2.INTER_AREA)
        mask = cv2.resize(((~sky) & (tree < 0.15) & (~exclude)).astype(np.uint8),
                          (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = cv2.erode(mask, np.ones((5, 5), np.uint8)) * 255
        kp, des = cv2.SIFT_create(nfeatures=nfeatures).detectAndCompute(
            cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), mask)
        scale = np.array([w / small.shape[1], h / small.shape[0]])
        xy = (np.array([k.pt for k in kp]).reshape(-1, 2) + 0.5) * scale - 0.5
        result[key] = dict(xy=xy, descriptors=des if des is not None else np.empty((0, 128), np.float32),
                           shape=np.array([h, w]), horizon=horizon, signature=signature)
        np.savez_compressed(path, **result[key])
        print(f'{key}: {len(xy)} SIFT features, {len(horizon)} horizon samples', flush=True)
        del rgb, sky, tree, small, mask
    return result


def verify_pair(x, y, threshold=4.):
    """Choose homography for rotation/planarity, fundamental matrix for parallax.

    These are outlier filters, not the bundle adjustment measurement model.
    """
    if len(x) < 12:
        return np.zeros(len(x), bool), 'insufficient'
    _, hm = cv2.findHomography(x, y, cv2.USAC_MAGSAC, threshold)
    _, fm = cv2.findFundamentalMat(x, y, cv2.USAC_MAGSAC, threshold, 0.999, 10000)
    hm = np.zeros(len(x), bool) if hm is None else hm.ravel().astype(bool)
    fm = np.zeros(len(x), bool) if fm is None else fm.ravel().astype(bool)
    if hm.sum() >= max(12, 0.85 * fm.sum()):
        return hm, 'homography'
    return fm, 'fundamental'


def match_features(features, per_pair=60, ratio=0.75):
    """Mutual ratio-tested SIFT matches, verified independently of fitted poses."""
    bf = cv2.BFMatcher(cv2.NORM_L2)
    pairs, diagnostics = [], []
    cv2.setRNGSeed(42)
    for a, b in combinations(features, 2):
        fa, fb = features[a], features[b]
        if min(len(fa['xy']), len(fb['xy'])) < 2:
            continue
        def good(d1, d2):
            return {m.queryIdx: m for v in bf.knnMatch(d1, d2, k=2)
                    if len(v) == 2 for m, n in [v] if m.distance < ratio * n.distance}
        ab = good(fa['descriptors'], fb['descriptors'])
        ba = good(fb['descriptors'], fa['descriptors'])
        match = [m for i, m in ab.items() if m.trainIdx in ba and ba[m.trainIdx].trainIdx == i]
        ids = np.array([(m.queryIdx, m.trainIdx) for m in match], dtype=int).reshape(-1, 2)
        keep, model = verify_pair(fa['xy'][ids[:, 0]], fb['xy'][ids[:, 1]])
        ids = ids[keep]
        diagnostics.append(dict(a=a, b=b, mutual=len(match), inliers=len(ids), model=model))
        if len(ids) < 12:
            continue
        # Spread points over the source frame, avoiding domination by one rock.
        xy = fa['xy'][ids[:, 0]]
        cells = np.floor(xy / 180).astype(int)
        _, selected = np.unique(cells, axis=0, return_index=True)
        ids = ids[np.sort(selected)]
        if len(ids) > per_pair:
            ids = ids[np.linspace(0, len(ids) - 1, per_pair).astype(int)]
        pairs.append((a, b, ids))
        print(f'{a}-{b}: {keep.sum()}/{len(match)} {model} inliers, using {len(ids)}', flush=True)
    return pairs, diagnostics


def build_tracks(pairs):
    """Merge shared feature IDs across pairs; reject conflicting track unions."""
    parent, members = {}, {}
    def find(x):
        if x not in parent:
            parent[x], members[x] = x, {x[0]: x[1]}
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    conflicts = 0
    for a, b, ids in pairs:
        for ia, ib in ids:
            ra, rb = find((a, int(ia))), find((b, int(ib)))
            if ra == rb:
                continue
            if members[ra].keys() & members[rb].keys():
                conflicts += 1
                continue
            parent[rb] = ra
            members[ra].update(members.pop(rb))
    tracks = [list(m.items()) for m in members.values() if len(m) >= 2]
    return sorted(tracks, key=len, reverse=True), conflicts


@dataclass
class Settings:
    tie_sigma_px: float = 4.
    antenna_sigma_px: float = 3.
    horizon_sigma_px: float = 10.
    terrain_sigma_m: float = 5.
    gps_floor_m: float = 15.
    focal_log_sigma: float = 0.15
    skyline_samples: int = 384


class Bundle:
    """Sparse joint fit of camera poses, 3D tracks, and ONE unknown antenna."""
    def __init__(self, poses, antenna, features, tracks, terrain, meta, gps,
                 settings=None):
        self.keys = list(poses)
        self.features, self.terrain = features, terrain
        self.settings = settings or Settings()
        self.shapes = [tuple(features[k]['shape']) for k in self.keys]
        self.base = np.array([[poses[k][p] for p in PRM_ORDER] for k in self.keys])
        self.antenna0 = np.asarray(antenna, float)
        self.meta = meta
        self.horizons = [features[k]['horizon'] for k in self.keys]
        self.gps = gps
        # Initialize terrain tracks from all available DEM intersections and
        # positive-depth triangulation; choose the best all-view reprojection.
        hits = {}
        for i, k in enumerate(self.keys):
            ids = sorted({fid for t in tracks for key, fid in t if key == k})
            if ids:
                xyz = terrain.intersect(self.base[i], self.shapes[i], features[k]['xy'][ids])
                hits.update({(k, fid): point for fid, point in zip(ids, xyz)})
        points, accepted, obs_cam, obs_point, obs_xy = [], [], [], [], []
        for track in tracks:
            track = [(k, fid) for k, fid in track if k in self.keys]
            if len(track) < 2:
                continue
            ci = [self.keys.index(k) for k, _ in track]
            xy = [features[k]['xy'][fid] for k, fid in track]
            origins = self.base[ci, :3]
            directions = np.array([rays(self.base[i], self.shapes[i], q)[0] for i, q in zip(ci, xy)])
            mat = np.eye(3)[None] - directions[:, :, None] * directions[:, None, :]
            candidates = [hits[(k, fid)] for k, fid in track if np.all(np.isfinite(hits[(k, fid)]))]
            if np.linalg.cond(mat.sum(axis=0)) < 1e6:
                candidates.append(np.linalg.solve(mat.sum(axis=0), np.einsum('nij,nj->i', mat, origins)))
            best = None
            for point in candidates:
                h = terrain.height(point[0], point[1])
                if not np.isfinite(h) or abs(point[2] - h) > 100:
                    continue
                pred_z = [project(self.base[i], self.shapes[i], point) for i in ci]
                if any(z[0] < 1 for _, z in pred_z):
                    continue
                score = np.mean([np.sum((v[0] - q)**2) for (v, _), q in zip(pred_z, xy)])
                if best is None or score < best[0]:
                    best = score, point
            if best is None:
                continue
            j = len(points)
            points.append(best[1])
            accepted.append(track)
            obs_cam.extend(ci)
            obs_point.extend([j] * len(ci))
            obs_xy.extend(xy)
        if not points:
            raise ValueError('No valid terrain tracks; inspect matching and seed poses.')
        self.tracks = accepted
        self.points0 = np.array(points)
        self.obs_cam, self.obs_point = np.array(obs_cam), np.array(obs_point)
        self.obs_xy = np.array(obs_xy)
        self.ant_cam = np.array([i for i, k in enumerate(self.keys) if 'ant_px' in meta.get(k, {})])
        self.ant_xy = np.array([meta[self.keys[i]]['ant_px'] for i in self.ant_cam])
        if len(self.ant_cam) < 2:
            raise ValueError('At least two labeled views are needed for antenna localization.')
        self.ncam, self.npoint = len(self.keys), len(points)
        # Optimize offsets in physical units; focal length uses a log ratio.
        self.x0 = np.zeros(7 * self.ncam + 3 * self.npoint + 3)
        self.scale = np.r_[np.tile([10, 10, 10, .03, .03, .02, .05], self.ncam),
                           np.full(3 * self.npoint + 3, 10.)]
        self.lower = np.r_[np.tile([-100, -100, -100, -.7, -.7, -.4, -.5], self.ncam),
                           np.full(3 * self.npoint, -500.), [-150, -150, -150]]
        self.upper = -self.lower
        # Keep all DEM evaluations in-bounds (initial point/camera is already inside).
        for i, p in enumerate(self.base):
            self.lower[7*i:7*i+2] = np.maximum(self.lower[7*i:7*i+2], [terrain.e[0]+2-p[0], terrain.n[0]+2-p[1]])
            self.upper[7*i:7*i+2] = np.minimum(self.upper[7*i:7*i+2], [terrain.e[-1]-2-p[0], terrain.n[-1]-2-p[1]])
        for j, p in enumerate(self.points0):
            off = 7*self.ncam + 3*j
            self.lower[off:off+2] = np.maximum(self.lower[off:off+2], [terrain.e[0]+1-p[0], terrain.n[0]+1-p[1]])
            self.upper[off:off+2] = np.minimum(self.upper[off:off+2], [terrain.e[-1]-1-p[0], terrain.n[-1]-1-p[1]])
        self.sparsity = self._sparsity()

    def unpack(self, x):
        delta = np.asarray(x[:7*self.ncam]).reshape(-1, 7)
        cams = self.base + delta
        cams[:, 6] = self.base[:, 6] * np.exp(delta[:, 6])
        points = self.points0 + x[7*self.ncam:-3].reshape(-1, 3)
        return cams, points, self.antenna0 + x[-3:]

    def components(self, x):
        cams, points, antenna = self.unpack(x)
        pred, depth = np.empty_like(self.obs_xy), np.empty(len(self.obs_xy))
        horizon, gps, focal, below = [], [], [], []
        for i, (p, shape, boundary) in enumerate(zip(cams, self.shapes, self.horizons)):
            use = self.obs_cam == i
            pred[use], depth[use] = project(p, shape, points[self.obs_point[use]])
            d = rays(p, shape, boundary)
            az = np.arctan2(d[:, 1], d[:, 0])
            observed = np.arctan2(d[:, 2], np.hypot(d[:, 0], d[:, 1]))
            modeled = self.terrain.skyline(p[:3], az, self.settings.skyline_samples)
            horizon.append((observed - modeled) * self.base[i, 6])
            entry = self.gps.get(self.keys[i])
            if entry is not None:
                gps.extend((p[:2] - entry[:2]) / max(entry[2], self.settings.gps_floor_m))
            focal.append(np.log(p[6] / self.base[i, 6]) / self.settings.focal_log_sigma)
            below.append(max(0., float(self.terrain.height(p[0], p[1])) + .1 - p[2]) / 3.)
        ant_pred, ant_depth = zip(*(project(cams[i], self.shapes[i], antenna) for i in self.ant_cam))
        return dict(tie_px=pred-self.obs_xy, tie_depth=depth,
                    antenna_px=np.concatenate(ant_pred)-self.ant_xy,
                    antenna_depth=np.concatenate(ant_depth), horizon_px=np.concatenate(horizon),
                    terrain_m=points[:, 2]-self.terrain.height(points[:, 0], points[:, 1]),
                    gps=np.asarray(gps), focal=np.asarray(focal), below=np.asarray(below))

    def residuals(self, x):
        c, s = self.components(x), self.settings
        return np.r_[c['tie_px'].ravel()/s.tie_sigma_px,
                     np.minimum(c['tie_depth']-1., 0.) / .1,
                     c['antenna_px'].ravel()/s.antenna_sigma_px,
                     np.minimum(c['antenna_depth']-1., 0.) / .1,
                     c['horizon_px']/s.horizon_sigma_px,
                     c['terrain_m']/s.terrain_sigma_m, c['gps'], c['focal'], c['below']]

    def _sparsity(self):
        """Exact residual dependency structure for grouped finite differences."""
        rows = []
        camera = lambda i: list(range(7*i, 7*i+7))
        point = lambda j: list(range(7*self.ncam+3*j, 7*self.ncam+3*j+3))
        ant = list(range(len(self.x0)-3, len(self.x0)))
        for i, j in zip(self.obs_cam, self.obs_point):
            rows.extend([camera(i)+point(j)]*2)
        rows.extend(camera(i)+point(j) for i, j in zip(self.obs_cam, self.obs_point))
        for i in self.ant_cam:
            rows.extend([camera(i)+ant]*2)
        rows.extend(camera(i)+ant for i in self.ant_cam)
        for i, boundary in enumerate(self.horizons):
            rows.extend([camera(i)]*len(boundary))
        rows.extend(point(j) for j in range(self.npoint))
        for i, k in enumerate(self.keys):
            if k in self.gps:
                rows.extend([camera(i)]*2)
        rows.extend(camera(i) for i in range(self.ncam))
        rows.extend(camera(i) for i in range(self.ncam))
        mat = lil_matrix((len(rows), len(self.x0)), dtype=int)
        for r, cols in enumerate(rows):
            mat[r, cols] = 1
        return mat.tocsr()

    def loss(self, z):
        # Manual antenna labels must retain quadratic influence: robustifying
        # them lets thousands of terrain features dismiss a visible antenna miss.
        # Feature outliers and DEM/skyline mismatch still use soft-L1.
        nobs, nant = len(self.obs_xy), len(self.ant_cam)
        robust = np.zeros(self.sparsity.shape[0], dtype=bool)
        robust[:2*nobs] = True
        h0 = 3*nobs + 3*nant
        robust[h0:h0 + sum(map(len, self.horizons)) + self.npoint] = True
        rho = np.array([z, np.ones_like(z), np.zeros_like(z)])
        t = 1 + z[robust]
        rho[:, robust] = [2*(np.sqrt(t)-1), 1/np.sqrt(t), -.5*t**(-1.5)]
        return rho

    def solve(self, start=None, max_nfev=150):
        start = self.x0 if start is None else np.asarray(start)
        return least_squares(self.residuals, start, jac_sparsity=self.sparsity,
                             bounds=(self.lower, self.upper), x_scale=self.scale,
                             loss=self.loss, f_scale=2., max_nfev=max_nfev,
                             ftol=1e-5, xtol=1e-6, gtol=1e-5)

    def metrics(self, x):
        c = self.components(x)
        cams, _, ant = self.unpack(x)
        report = {}
        for i, key in enumerate(self.keys):
            use = self.obs_cam == i
            d = rays(cams[i], self.shapes[i], self.horizons[i])
            observed = np.arctan2(d[:, 2], np.hypot(d[:, 0], d[:, 1]))
            predicted = self.terrain.skyline(cams[i, :3], np.arctan2(d[:, 1], d[:, 0]), self.settings.skyline_samples)
            row = dict(tie_median_px=float(np.median(np.linalg.norm(c['tie_px'][use], axis=1))) if use.any() else None,
                       observations=int(use.sum()),
                       horizon_angular_px_rms=float(np.sqrt(np.mean(((observed-predicted)*self.base[i, 6])**2))) if len(d) else None,
                       camera_height_m=float(cams[i, 2]-self.terrain.height(*cams[i, :2])))
            if i in self.ant_cam:
                q, z = project(cams[i], self.shapes[i], ant)
                row['antenna_px'] = float(np.linalg.norm(q[0]-self.meta[key]['ant_px']))
                row['antenna_in_front'] = bool(z[0] > 0)
            report[key] = row
        residual = self.residuals(x)
        return dict(antenna_enu=ant.tolist(), images=report,
                    robust_cost=float(2*np.sum(self.loss((residual/2)**2)[0])),
                    tie_median_px=float(np.median(np.linalg.norm(c['tie_px'], axis=1))),
                    terrain_abs_median_m=float(np.median(abs(c['terrain_m']))),
                    behind_camera=int((c['tie_depth'] <= 0).sum()))


def gps_from_cache(path):
    with np.load(path) as z:
        return {str(k): np.array([e, n, err]) for k, e, n, err in
                zip(z['keys'], z['e_gps'], z['n_gps'], z['h_err_m']) if np.all(np.isfinite([e, n, err]))}


def mcmc_initvals(poses, antenna, dem):
    """Convert a CV candidate to PyMC initvals without changing any priors.

    Use the SAME DEM and explicit image order as the posterior model. Refuse
    below-ground starts instead of silently clipping them to 1 mm clearance.
    """
    values = {}
    for key, p in poses.items():
        for name in PRM_ORDER:
            if name != 'u':
                values[f'{key}_{name}'] = float(p[name])
        h = p['u'] - float(dem.interp_alt(p['e'],p['n']))
        if not np.isfinite(h) or h <= 0:
            raise ValueError(f'{key}: camera is not above the posterior model DEM (h={h})')
        values[f'{key}_log_h'] = float(np.log(h))
    e, n, u = antenna
    h = u - float(dem.interp_alt(e,n))
    if not np.isfinite(h) or h <= 0:
        raise ValueError(f'Antenna is not above the posterior model DEM (h={h})')
    values.update(ant_e=float(e), ant_n=float(n), ant_log_h=float(np.log(h)))
    return values


def run(root=Path('.'), fit='fit_bundle_v5_2231_polished.npz',
        dem_file='marjum_dem_sw.npz', output='cv_initialization', starts=2,
        max_nfev=150, keys=None, matcher='sift', settings=None):
    from eigsep_terrain.marjum_dem import MarjumDEM
    from eigsep_terrain.fitio import load_fit, save_fit
    root = Path(root)
    out = root / output
    out.mkdir(parents=True, exist_ok=True)
    # Never overwrite a completed experiment accidentally.
    if (out / 'report.json').exists():
        raise FileExistsError(f'{out}/report.json exists; choose a new output directory')
    poses, antenna, _ = load_fit(root / fit)
    if keys:
        poses = {k: poses[k] for k in keys}
    if matcher == 'sift':
        features = extract_features(root, list(poses), root/'cv_features')
        pairs, pair_report = match_features(features)
    elif matcher == 'lightglue':
        from marjum_lightglue import learned_matches
        features, pairs, pair_report = learned_matches(root, list(poses))
    else:
        raise ValueError(f'Unknown matcher: {matcher}')
    tracks, conflicts = build_tracks(pairs)
    print(f'{len(tracks)} tracks; {conflicts} conflicting merges rejected', flush=True)
    terrain = Terrain(MarjumDEM(cache_file=str(root/dem_file)))
    meta = json.loads((root/'meta.json').read_text())
    bundle = Bundle(poses, antenna, features, tracks, terrain, meta,
                    gps_from_cache(root/'marjum_2026_07_exif.npz'), settings=settings)
    print(f'{bundle.ncam} cameras, {bundle.npoint} initialized tracks, {len(bundle.obs_xy)} observations', flush=True)
    before = bundle.metrics(bundle.x0)
    best, candidates = None, []
    rng = np.random.default_rng(42)
    for trial in range(starts):
        x = bundle.x0.copy()
        if trial:
            # Independent starts around the original state, not jittering a winner.
            x[:7*bundle.ncam] = (rng.normal(size=7*bundle.ncam)*bundle.scale[:7*bundle.ncam]*.5)
            x[-3:] = rng.normal(size=3)*5.
            x = np.clip(x, bundle.lower+1e-7, bundle.upper-1e-7)
        print(f'start {trial+1}/{starts}: optimizing', flush=True)
        result = bundle.solve(x, max_nfev=max_nfev)
        metrics = bundle.metrics(result.x)
        candidates.append(dict(start=trial, success=bool(result.success), message=result.message,
                               nfev=result.nfev, **metrics))
        cams, points, ant = bundle.unpack(result.x)
        save_fit(out/f'candidate_{trial}.npz', {k:dict(zip(PRM_ORDER,p)) for k,p in zip(bundle.keys,cams)}, ant)
        np.savez_compressed(out/f'state_{trial}.npz', x=result.x, points=points,
                            obs_cam=bundle.obs_cam, obs_point=bundle.obs_point, obs_xy=bundle.obs_xy,
                            shapes=np.array(bundle.shapes), keys=np.array(bundle.keys))
        print(f'start {trial}: cost={metrics["robust_cost"]:.1f}, tie median={metrics["tie_median_px"]:.2f}px, antenna={ant}', flush=True)
        if best is None or result.cost < best.cost:
            best = result
    cams, points, ant = bundle.unpack(best.x)
    save_fit(out/'fit_cv.npz', {k:dict(zip(PRM_ORDER,p)) for k,p in zip(bundle.keys,cams)}, ant)
    report = dict(input_fit=fit, dem=dem_file, matcher=matcher, settings=asdict(bundle.settings),
                  tracks=bundle.npoint, observations=len(bundle.obs_xy), conflicting_merges=conflicts,
                  pairs=pair_report, before=before, after=bundle.metrics(best.x), candidates=candidates,
                  caveat='Initialization objective only; residual scales are working assumptions, not calibrated uncertainties.')
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    return bundle, best, report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit', default='fit_bundle_v5_2231_polished.npz')
    parser.add_argument('--dem-file', default='marjum_dem_sw.npz')
    parser.add_argument('--output', default='cv_initialization')
    parser.add_argument('--starts', type=int, default=2)
    parser.add_argument('--max-nfev', type=int, default=150)
    parser.add_argument('--keys', nargs='+')
    parser.add_argument('--matcher', choices=['sift','lightglue'], default='sift')
    args = parser.parse_args()
    if args.starts < 1:
        parser.error('--starts must be positive')
    run(**vars(args))
