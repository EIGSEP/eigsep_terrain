import os
import numpy as np
from .imageio import load_image
from .utils import rot_m, mask_near_horizon, fill_psky_holes
from .ray_numba import ray_distance_coarse_to_fine_numba
from .tiepoints import project_via_pose, project_world_point, _project_direction
import cv2
# .seg (torch + transformers) and pymc are imported lazily, inside the
# specific methods that need them (segment_image, PositionSolver.
# get_mcmc_prms) -- both are heavy imports with a real memory cost, and
# most uses of this module (pose refinement, tie points, antenna
# reprojection) need neither, especially once an image's segmentation is
# already cached.

PRM_ORDER = ('e', 'n', 'u', 'th', 'ph', 'ti', 'f')
dtype_r = np.float32

def pixels_to_rays(Nu, Nv, f, uv=None, dtype=dtype_r):
    if uv is None:
        _u = np.arange(Nu, dtype=int)
        _v = np.arange(Nv, dtype=int)
        uv = np.meshgrid(_v, _u)[::-1]
    u, v = uv
    rays = np.array([Nu // 2 - u, Nv // 2 - v, np.full(u.shape, f)], dtype=dtype)
    rays /= np.linalg.norm(rays, axis=0)
    return rays


class HorizonImage:
    def __init__(self, filename, meta=None, **kwargs):
        if meta is None:
            meta = {}
        self.filename = filename
        base, _ext = os.path.splitext(os.path.basename(filename))
        self.key = base.split('_')[-1]
        self.npzfile = 'img_seg_' + base + '.npz'
        self.posefile = 'pose_' + base + '.npz'
        self.img = np.flipud(load_image(self.filename))
        self.px_dist = kwargs.pop('px_dist', 150)  # px_dist from mask_near_horizon
        self.px_smooth = kwargs.pop('px_smooth', 100)  # px_dist from mask_near_horizon

        if not os.path.exists(self.npzfile):
            segdict = self.segment_image()
            self.save_segment_image(segdict)
        self.sky_mask, _psky, self.ptree = self.read_psky()
        _hmask, _ = self.gen_horizon_mask(px_dist=150)  # XXX manual tuned px_dist
        maybe_tree = _hmask * self.ptree
        psky = np.where(maybe_tree > 0.05, 0.5, _psky)  # XXX manual thresh
        sky, psky_filled = fill_psky_holes(psky, 0.6, 200**2, 8, 50)
        ker = (self.px_smooth, self.px_smooth)
        psky_blur = cv2.blur(psky_filled, ker, cv2.BORDER_DEFAULT)
        self.psky = psky_blur
        self.horizon_mask, self.horizon_dist = self.gen_horizon_mask()
        
        if self.key in meta:
            self.meta = meta[self.key]
            self.set_prms(self.meta.get('prms', [0.0 for _ in PRM_ORDER]))
        else:
            self.set_prms([0.0 for k in PRM_ORDER])
        self._px_choice = None

    def set_prms(self, prms):
        self.prms = dict(zip(PRM_ORDER, prms))

    def get_prms(self):
        return (self.prms[k] for k in PRM_ORDER)

    @property
    def prms_str(self):
        return f"{self.prms['e']: 7.2f}, {self.prms['n']: 7.2f}, {self.prms['u']: 7.2f}, {self.prms['th']: 6.4f}, {self.prms['ph']: 6.4f}, {self.prms['ti']: 5.4f}, {self.prms['f']: 7.2f}"

    @property
    def npix_y(self):
        return self.img.shape[0]
        
    @property
    def npix_x(self):
        return self.img.shape[1]
        
    def segment_image(self, device='cpu', thr=0.6, fill_thresh=200**2,
                      connectivity=8, px_dist=150):
        # lazy: seg.py pulls in torch + transformers, which cost real memory
        # to even import -- avoid paying that unless a fresh segmentation is
        # actually needed (the common case is a cached img_seg_*.npz already
        # existing, e.g. every image used in this project's later phases)
        from .seg import TiledSkyProbSegFormer
        seg = TiledSkyProbSegFormer(device=device)
        _psky, _ptree = seg.p_sky_tiled(self.filename, tile=1024, overlap=256, batch=2)
        self.sky_mask, _ = fill_psky_holes(_psky, thr, fill_thresh,
                                           connectivity, px_dist)
        return {'skymask': self.sky_mask, 'psky': _psky, 'ptree': _ptree}

    def save_segment_image(self, segdict):
        np.savez(self.npzfile, **segdict)

    def read_psky(self):
        npz = np.load(self.npzfile)
        psky = np.flipud(npz['psky'])
        ptree = np.flipud(npz['ptree'])
        skymask = np.flipud(npz['skymask'])
        return skymask, psky, ptree
        
    def get_rays(self, pixels=None, dtype=dtype_r):
        z_rays = pixels_to_rays(self.npix_y, self.npix_x,
                                f=self.prms['f'], uv=pixels, dtype=dtype)
        rm_tilt = rot_m(self.prms['ti'], np.array([0,0,1], dtype=dtype))
        rm_th   = rot_m(self.prms['th'], np.array([0,1,0], dtype=dtype))
        rm_ph   = rot_m(self.prms['ph'], np.array([0,0,1], dtype=dtype))
        rm = rm_ph @ (rm_th @ rm_tilt)
        rays = np.einsum('ij,j...->i...', rm, z_rays)
        return rays

    def choose_pixels(self, N=1000, mask=None, reset=False):
        if reset:
            self._px_choice = None
        if self._px_choice is None:
            if mask is None:
                mask = self.horizon_mask
            x, y = np.where(mask)
            w = np.exp(-0.5 * self.horizon_dist[x, y]**2 / (self.px_dist / 2)**2)
            rng = np.random.default_rng()
            inds = rng.choice(x.size, size=N, replace=False, p=w / w.sum())
            self._px_choice = (x[inds], y[inds])
        return self._px_choice

    def ray_distance(self, dem, rays, dtype=dtype_r):
        rays_2d = rays.reshape(rays.shape[0], -1)
        (E, N), U = dem.get_en(), dem.data
        start_point = np.array([self.prms[k] for k in 'enu'], dtype=dtype)
        r = ray_distance_coarse_to_fine_numba(E, N, U, start_point, rays_2d)
        r.shape = rays.shape[1:]
        return r

    def gen_horizon_mask(self, px_dist=None):
        if px_dist is None:
            px_dist = self.px_dist
        horizon_mask, horizon_dist = mask_near_horizon(self.sky_mask, px_dist)
        return horizon_mask, horizon_dist

    def export_jax(self, n_rays=1000, eps=1e-3, dtype=dtype_r):
        x_px, y_px = self.choose_pixels(N=n_rays)
        psky = self.psky[x_px, y_px].astype(dtype).clip(eps, 1-eps)
        ant_px = np.array(self.meta['ant_px'][::-1], dtype=np.int32)
        return dict(
            key=self.key,
            npix_y=np.int32(self.npix_y),
            npix_x=np.int32(self.npix_x),
            x_px=x_px.astype(np.int32),
            y_px=y_px.astype(np.int32),
            psky=psky,
            ant_px=ant_px,
        )
        
    def horizon_ray_logL(self, dem, n_rays=1000, dtype=dtype_r, eps=1e-3,
                         correlation_px=None):
        x_px, y_px = self.choose_pixels(N=n_rays)
        # Per-pixel probability that the pixel is sky
        psky = self.psky[x_px, y_px].clip(eps, 1 - eps) # Avoid log(0)

        # Evaluate your geometric horizon model (binary)
        rays = self.get_rays(pixels=(x_px, y_px), dtype=dtype)
        r = self.ray_distance(dem, rays, dtype=dtype)
        model_sky = np.isnan(r)  # True => model predicts sky

        # if model says sky, probability of observing "sky" is psky, else 1-psky
        logp_sky = np.log(psky)
        logp_ground = np.log1p(-psky)  # stable log(1-psky)
        per_pixel = np.where(model_sky, logp_sky, logp_ground)

        # The n_rays samples are not independent evidence: psky itself is
        # smoothed with a (px_smooth, px_smooth) box blur (see __init__), so
        # any two samples within about px_smooth pixels of each other carry
        # nearly identical information. Summing n_rays raw per-pixel terms
        # as if they were independent overstates this likelihood's
        # statistical weight by a large factor (n_rays can be thousands,
        # true independent information is capped by how many
        # px_smooth-sized patches the sampled region actually spans) --
        # badly enough that genuinely independent measurements (GPS, tie
        # points) can never meaningfully compete against it. Rescale the
        # mean per-pixel logL by an effective sample count instead of
        # summing the raw count, preserving the likelihood's shape (still
        # driven by the same fraction of correct/incorrect classification)
        # while keeping its magnitude honest relative to other terms.
        if correlation_px is None:
            correlation_px = self.px_smooth
        col_span = max(y_px.max() - y_px.min(), 1)
        n_eff = np.clip(col_span / correlation_px, 1, n_rays)
        return np.mean(per_pixel) * n_eff

    def _raster_boundary(self, sky2d):
        '''Per-column index of the topmost ground row in a boolean sky
        array (True=sky), NaN where a column is all-sky.'''
        H, _W = sky2d.shape
        ridx = np.arange(H)[:, None]
        idx = np.where(~sky2d, ridx, -1).max(axis=0)
        return np.where(idx >= 0, idx.astype(float), np.nan)

    def refine_pose(self, dem, free_keys=('e', 'n', 'u', 'th', 'ph', 'ti'),
                     decimate_coarse=12, decimate_fine=4,
                     th_range_deg=(-45, 21, 2.5), ph_range_deg=(-45, 46, 2.5),
                     cache_file=None, overwrite=False,
                     ant_pos=None, ant_sigma_m=0.5,
                     tie_targets=None, tie_sigma_px=10.0,
                     landmark_targets=None, landmark_sigma_px=10.0):
        '''Fit self.prms to the segmented horizon by matching the ray-traced
        sky/ground boundary curve against the segmentation's, column by
        column -- not a sparse per-pixel likelihood, which in practice
        proved too noisy/prone to bad local optima for this (real ridgelines
        include tree canopy a bare-earth DEM can't reproduce, and sparse
        sampling amplifies that).

        Stage 1: a coarse grid search over elevation/azimuth offsets on a
        heavily decimated raster, since the boundary-match objective is
        flat/saturated far from the true pose (e.g. every ray escapes to
        sky regardless of azimuth), which traps a local optimizer started
        cold. Stage 2: a joint Nelder-Mead polish of free_keys (letting
        position float, not just angles -- single-image fits have a
        real position/orientation degeneracy that a coordinate-wise search
        cannot escape) on a finer decimated raster.

        If ant_pos (a fixed world ENU point, e.g. a running antenna-
        position estimate) is given and self.meta['ant_px'] exists, the
        squared pixel distance between ant_px and ant_pos's predicted
        projection (eigsep_terrain.tiepoints.project_world_point) is added
        to both stages' objectives, weighted by 1/sigma_px**2 where
        sigma_px = f * ant_sigma_m / distance-to-antenna -- i.e. ant_sigma_m
        is a *physical* antenna-pick uncertainty in meters (same idea as
        ant_logL's box_size), converted to this image's own pixel scale via
        its focal length and current camera-to-antenna distance, rather
        than an arbitrary dimensionless pixel weight. A flat pixel weight
        (tried first) turned out to punish some images far more than
        others for the same real-world click precision, since a pixel
        error means a very different angular error depending on focal
        length and distance -- exactly what this recalibration fixes. This
        is what lets a single image's pose be pulled into agreement with a
        shared antenna position via ordinary deterministic optimization
        instead of MCMC (see notebook history for the "one image at a
        time" bundle-adjustment-style refinement this enables).

        tie_targets, if given, is a list of (other_img, pts_self_xy,
        pts_other_xy) tuples -- other_img a HorizonImage held fixed,
        pts_self_xy/pts_other_xy matched (col, row) point arrays between
        self and other_img (e.g. from eigsep_terrain.tiepoints.
        match_tie_points, restricted to inliers). Each target contributes
        mean squared pixel residual (self's points projected into
        other_img's frame via project_via_pose, vs. the actual match)
        divided by tie_sigma_px**2, averaged across targets and added to
        both stages' objectives alongside the antenna term -- the same
        mechanism, just against another image's fixed pose instead of a
        3D point. tie_sigma_px defaults to a flat pixel scale (unlike
        ant_sigma_m) since SIFT/RANSAC matches don't have a comparable
        physical size to convert from; 10 px reflects it being tighter
        than a manual antenna click but not so tight that a few points on
        tree canopy (which a bare-earth DEM was already found unable to
        match -- see refine_pose's own module docstring history) can
        dominate the fit.

        landmark_targets, if given, is a list of (point_enu, row, col)
        tuples -- point_enu a fixed 3D world point (e.g. where another,
        trusted image's ray hits the DEM -- see
        eigsep_terrain.tiepoints.project_world_point for the reverse
        direction), row/col self's pixel coordinate that should see it.
        Unlike tie_targets (which assumes negligible parallax between the
        two cameras, fine for a small baseline but wrong once the cameras
        are tens of meters apart), this uses self's actual camera position
        via project_world_point, so it's exact regardless of baseline --
        the right tool once two images share a real, non-negligible
        parallax baseline. Contributes mean squared pixel residual over
        all targets divided by landmark_sigma_px**2 to both stages'
        objectives, same flat-pixel-scale reasoning as tie_sigma_px.

        The fit takes a few minutes per image, so the result is cached to
        cache_file (default: self.posefile, i.e. 'pose_<basename>.npz') and
        reused on subsequent calls. Pass overwrite=True to force a fresh fit
        and refresh the cache (e.g. after changing free_keys/decimate/range/
        ant_pos arguments, which the cache does not itself track).

        Updates self.prms to the optimum and returns (prms_dict, info).
        info is the scipy OptimizeResult from a fresh fit (with an added
        `.from_cache = False` attribute), or a lightweight stand-in with
        `.success`, `.fun` (cached SSD), and `.from_cache = True` when
        loaded from cache.'''
        from scipy.optimize import minimize
        from types import SimpleNamespace

        if cache_file is None:
            cache_file = self.posefile
        if os.path.exists(cache_file) and not overwrite:
            npz = np.load(cache_file)
            prms_ref = {k: float(npz[k]) for k in PRM_ORDER}
            self.set_prms([prms_ref[k] for k in PRM_ORDER])
            info = SimpleNamespace(success=True, fun=float(npz['ssd']),
                                    nfev=0, from_cache=True)
            return prms_ref, info

        def ant_sq_resid():
            if ant_pos is None or not hasattr(self, 'meta') or 'ant_px' not in self.meta:
                return 0.0
            row_pred, col_pred = project_world_point(ant_pos, self)
            ax, ay = self.meta['ant_px']
            cam = np.array([self.prms['e'], self.prms['n'], self.prms['u']])
            dist = np.linalg.norm(np.asarray(ant_pos, dtype=float) - cam)
            sigma_px = self.prms['f'] * ant_sigma_m / dist
            return ((col_pred - ax)**2 + (row_pred - ay)**2) / sigma_px**2

        def tie_sq_resid():
            if not tie_targets:
                return 0.0
            total = 0.0
            for other_img, pts_self_xy, pts_other_xy in tie_targets:
                rows_self, cols_self = pts_self_xy[:, 1], pts_self_xy[:, 0]
                row_pred, col_pred = project_via_pose(self, (rows_self, cols_self), other_img)
                d2 = (col_pred - pts_other_xy[:, 0])**2 + (row_pred - pts_other_xy[:, 1])**2
                total += np.mean(d2) / tie_sigma_px**2
            return total / len(tie_targets)

        def landmark_sq_resid():
            if not landmark_targets:
                return 0.0
            pts = np.asarray([t[0] for t in landmark_targets], dtype=float)
            rows = np.array([t[1] for t in landmark_targets], dtype=float)
            cols = np.array([t[2] for t in landmark_targets], dtype=float)
            cam = np.array([self.prms['e'], self.prms['n'], self.prms['u']])
            d = pts - cam
            d = d / np.linalg.norm(d, axis=1, keepdims=True)
            row_pred, col_pred = _project_direction(d.T, self)
            d2 = (col_pred - cols)**2 + (row_pred - rows)**2
            return np.mean(d2) / landmark_sigma_px**2

        base = dict(self.prms)

        # Build ray directions directly at the decimated pixel coordinates
        # (via pixels=) rather than computing the full npix_y x npix_x grid
        # and slicing afterward -- get_rays()[..., ::dec, ::dec] still pays
        # for the full-resolution rotation/normalization on every call
        # regardless of dec, which made decimation not actually save any
        # work (measured ~5x slower than building the decimated grid
        # directly).
        def pixel_grid(dec):
            rr, cc = np.meshgrid(np.arange(0, self.npix_y, dec),
                                  np.arange(0, self.npix_x, dec), indexing='ij')
            return rr, cc

        sl_c = slice(None, None, decimate_coarse)
        pix_c = pixel_grid(decimate_coarse)
        actual_c = self._raster_boundary(self.sky_mask[sl_c, sl_c])
        best = None
        for dth in np.deg2rad(np.arange(*th_range_deg)):
            for dph in np.deg2rad(np.arange(*ph_range_deg)):
                prms = dict(base)
                prms['th'] = base['th'] + dth
                prms['ph'] = base['ph'] + dph
                self.set_prms([prms[k] for k in PRM_ORDER])
                rays = self.get_rays(pixels=pix_c)
                r = self.ray_distance(dem, rays)
                pred = self._raster_boundary(np.isnan(r))
                m = ~np.isnan(actual_c) & ~np.isnan(pred)
                ssd = np.mean((actual_c[m] - pred[m])**2) if m.sum() >= 20 else np.inf
                ssd = ssd + ant_sq_resid() + tie_sq_resid() + landmark_sq_resid()
                if best is None or ssd < best[0]:
                    best = (ssd, dth, dph)
        d1 = dict(base)
        d1['th'] = base['th'] + best[1]
        d1['ph'] = base['ph'] + best[2]

        sl_f = slice(None, None, decimate_fine)
        pix_f = pixel_grid(decimate_fine)
        actual_f = self._raster_boundary(self.sky_mask[sl_f, sl_f])

        def objective(x):
            prms = dict(d1)
            for k, v in zip(free_keys, x):
                prms[k] = v
            self.set_prms([prms[k] for k in PRM_ORDER])
            rays = self.get_rays(pixels=pix_f)
            r = self.ray_distance(dem, rays)
            pred = self._raster_boundary(np.isnan(r))
            m = ~np.isnan(actual_f) & ~np.isnan(pred)
            if m.sum() < 20:
                return 1e6
            return np.mean((actual_f[m] - pred[m])**2) + ant_sq_resid() + tie_sq_resid() + landmark_sq_resid()

        x0 = np.array([d1[k] for k in free_keys], dtype=np.float64)
        default_steps = dict(e=3.0, n=3.0, u=3.0, th=np.deg2rad(1.0),
                              ph=np.deg2rad(1.0), ti=np.deg2rad(0.5),
                              f=d1['f'] * 0.05)
        steps = np.array([default_steps[k] for k in free_keys])
        initial_simplex = np.array(
            [x0] + [x0 + steps[i] * np.eye(len(x0))[i] for i in range(len(x0))]
        )
        res = minimize(objective, x0, method='Nelder-Mead',
                        options=dict(xatol=1e-4, fatol=1e-2, maxiter=3000,
                                     maxfev=6000,
                                     initial_simplex=initial_simplex))
        prms_ref = dict(d1)
        for k, v in zip(free_keys, res.x):
            prms_ref[k] = v
        self.set_prms([prms_ref[k] for k in PRM_ORDER])
        res.from_cache = False
        np.savez(cache_file, ssd=res.fun, **{k: prms_ref[k] for k in PRM_ORDER})
        return prms_ref, res

    def ant_logL(self, ant_pos, box_size):
        ant_ray = self.get_rays(np.array(self.meta['ant_px'][::-1]))
        r_ant = ant_pos - np.array([self.prms['e'], self.prms['n'], self.prms['u']])
        
        cos_pred = np.dot(ant_ray, r_ant) / (np.linalg.norm(ant_ray) * np.linalg.norm(r_ant))
        delta_theta = np.arccos(cos_pred.clip(-1, 1)) # rad
        sigma_theta = box_size / np.linalg.norm(r_ant)
        logL = np.log(1 / np.sqrt(2 * np.pi * sigma_theta**2)) - 0.5 * delta_theta**2 / sigma_theta**2
        return logL
    
class PositionSolver:
    def __init__(self, ant_pos_prior, fit_imgs, static_imgs, n_rays, dem,
                 ant_pos_err=20, box_size=0.3):
        self.fit_imgs = fit_imgs
        self.ant_pos_prior = ant_pos_prior
        self.ant_pos_err = ant_pos_err
        self.imgs = fit_imgs + static_imgs
        self.box_size = box_size
        self.dem = dem
        self.n_rays = n_rays
        self.tie_pairs = []
        self.gps_pairs = []

    def add_gps_relative_prior(self, img1, img2, gps_e1, gps_n1, gps_e2, gps_n2,
                                sigma=None, herr1=None, herr2=None):
        '''Register a soft prior on img1's position *relative to* img2's,
        from their EXIF GPS fixes (gps_e1, gps_n1) and (gps_e2, gps_n2) --
        not their absolute positions. Differencing out the absolute
        position this way means any common-mode GPS bias shared by both
        fixes cancels, so this only relies on GPS being informative about
        *relative* position between the two shots, which is a much weaker
        (and, checked empirically for this dataset, well-supported: photos
        seconds apart from a stationary burst agree to well under a meter)
        assumption than trusting either fix's absolute position, or than
        assuming the two images share the exact same camera position.

        sigma defaults to sqrt(herr1**2 + herr2**2) (each image's own
        GPSHPositioningError, combined in quadrature) if not given
        explicitly.'''
        if sigma is None:
            sigma = np.hypot(herr1, herr2)
        self.gps_pairs.append(dict(
            img1=img1, img2=img2,
            delta_e=gps_e1 - gps_e2, delta_n=gps_n1 - gps_n2,
            sigma=sigma,
        ))

    def gps_delta_logL(self):
        logL = 0.0
        for gp in self.gps_pairs:
            fit_de = gp['img1'].prms['e'] - gp['img2'].prms['e']
            fit_dn = gp['img1'].prms['n'] - gp['img2'].prms['n']
            r2 = (fit_de - gp['delta_e'])**2 + (fit_dn - gp['delta_n'])**2
            sigma2 = gp['sigma'] ** 2
            logL += -0.5 * r2 / sigma2 - np.log(2 * np.pi * sigma2)
        return logL

    def add_tie_points(self, img1, img2, pts1_xy, pts2_xy, sigma_px=5.0):
        '''Register tie points (e.g. from eigsep_terrain.tiepoints.
        match_tie_points) between img1 and img2 -- both should already be
        in self.imgs -- as an additional Gaussian likelihood term. At every
        total_logL evaluation, pts1_xy's ray is projected from img1's
        *current* pose into img2's frame and compared to pts2_xy (see
        eigsep_terrain.tiepoints.project_via_pose); this only holds for
        distant terrain and cameras close enough together that parallax is
        negligible, which is the regime tie points are useful in here.
        pts1_xy/pts2_xy are (N, 2) arrays in (col, row) OpenCV convention,
        matching match_tie_points' return. sigma_px sets how tightly the
        reprojection is enforced, in pixels.'''
        self.tie_pairs.append(dict(
            img1=img1, img2=img2,
            pts1=np.asarray(pts1_xy, dtype=dtype_r),
            pts2=np.asarray(pts2_xy, dtype=dtype_r),
            sigma_px=sigma_px,
        ))

    def tie_logL(self):
        logL = 0.0
        for tp in self.tie_pairs:
            rows1, cols1 = tp['pts1'][:, 1], tp['pts1'][:, 0]
            pred_row, pred_col = project_via_pose(tp['img1'], (rows1, cols1), tp['img2'])
            dr = pred_row - tp['pts2'][:, 1]
            dc = pred_col - tp['pts2'][:, 0]
            sigma2 = tp['sigma_px'] ** 2
            logL += np.sum(-0.5 * (dr**2 + dc**2) / sigma2 - np.log(2 * np.pi * sigma2))
        return logL

    def eval_cur_prms(self):
        prms = []
        for cnt, img in enumerate(self.fit_imgs):
            for k in PRM_ORDER:
                if k == 'u':
                    u0 = float(self.dem.interp_alt(
                        img.prms['e'], img.prms['n']
                    ))
                    h = img.prms['u'] - u0
                    prms.append(np.log(max(h, 1e-3)))
                else:
                    prms.append(img.prms[k])
        ant_e, ant_n, ant_u = self.ant_pos_prior
        ant_u0 = float(self.dem.interp_alt(ant_e, ant_n))
        ant_h = ant_u - ant_u0
        prms += [ant_e, ant_n, np.log(max(ant_h, 1e-3))]
        return prms

    def get_mcmc_prms(self):
        import pymc as pm
        prms = []
        for cnt, img in enumerate(self.fit_imgs):
            _sigmas = self.sigmas[
                cnt*len(PRM_ORDER): (cnt+1)*len(PRM_ORDER)
            ]
            for k, sig in zip(PRM_ORDER, _sigmas):
                if k == 'u':
                    u0 = float(self.dem.interp_alt(
                        img.prms['e'], img.prms['n']
                    ))
                    h = img.prms['u'] - u0
                    prms.append(pm.Normal(
                        f"{img.key}_log_h",
                        mu=np.log(max(h, 1e-3)),
                        sigma=sig,
                    ))
                else:
                    prms.append(
                        pm.Normal(f"{img.key}_{k}", mu=img.prms[k],
                                  sigma=sig)
                    )
        ant_e, ant_n, ant_u = self.ant_pos_prior
        ant_u0 = float(self.dem.interp_alt(ant_e, ant_n))
        ant_h = ant_u - ant_u0
        ant_sig_e, ant_sig_n, ant_sig_u = self.sigmas[-3:]
        prms += [
            pm.Normal('ant_e', mu=ant_e, sigma=ant_sig_e),
            pm.Normal('ant_n', mu=ant_n, sigma=ant_sig_n),
            pm.Normal('ant_log_h', mu=np.log(max(ant_h, 1e-3)),
                      sigma=ant_sig_u),
        ]
        return prms

    def set_mcmc_prms(self, theta_h):
        """Set parameters from theta, which uses h (height above ground)
        at the u position for each camera and for the antenna."""
        theta_u = self._convert_uh_prms(theta_h, sign=1)
        for cnt, img in enumerate(self.fit_imgs):
            img.set_prms(
                tuple(theta_u[cnt*len(PRM_ORDER):(cnt+1)*len(PRM_ORDER)])
            )
        self.ant_pos = np.asarray(theta_u[-3:])

    def prms_u_to_h(self, theta_u):
        """Convert a flat parameter vector from absolute-u to h
        (height above ground) at the u position for each camera and
        for the antenna. Returns a new float32 array."""
        return self._convert_uh_prms(theta_u, sign=-1)

    def _convert_uh_prms(self, theta_in, sign=1):
        """Convert between log_h and absolute-u representations.

        sign=+1 (set_mcmc_prms): theta_in has log_h; output has u = exp(log_h) + u0.
        sign=-1 (prms_u_to_h):   theta_in has u;     output has log_h = log(u - u0).
        """
        theta = np.array(theta_in, dtype=dtype_r)
        _ei = PRM_ORDER.index('e')
        _ni = PRM_ORDER.index('n')
        _ui = PRM_ORDER.index('u')
        for cnt in range(len(self.fit_imgs)):
            base = cnt * len(PRM_ORDER)
            e, n = theta[base + _ei], theta[base + _ni]
            u0 = float(self.dem.interp_alt(e, n))
            if sign == 1:
                theta[base + _ui] = np.exp(theta[base + _ui]) + u0
            else:
                theta[base + _ui] = np.log(
                    max(theta[base + _ui] - u0, dtype_r(1e-3))
                )
        ant_e, ant_n = theta[-3], theta[-2]
        ant_u0 = float(self.dem.interp_alt(ant_e, ant_n))
        if sign == 1:
            theta[-1] = np.exp(theta[-1]) + ant_u0
        else:
            theta[-1] = np.log(max(theta[-1] - ant_u0, dtype_r(1e-3)))
        return theta

    def set_mcmc_sigmas(self, pos_err=30.0, ang_err=np.deg2rad(5.0),
                        f_err=0.1, log_h_sigma=1.0):
        img_sigmas = (pos_err, pos_err, log_h_sigma,
                      ang_err, ang_err, ang_err, f_err)
        self.sigmas = [img.prms[k] * sig if k == 'f' else sig
                       for img in self.fit_imgs
                       for k, sig in zip(PRM_ORDER, img_sigmas)]
        self.sigmas += [pos_err, pos_err, log_h_sigma]

    @property
    def prms_str(self):
        imgs_str = [img.prms_str for img in self.fit_imgs]
        ant_str = f"{self.ant_pos[0]: 7.2f}, {self.ant_pos[1]: 7.2f}, {self.ant_pos[2]: 7.2f}"
        return ',\n'.join(imgs_str + [ant_str])

    def total_logL(self, theta, n_rays=None, eps=1e-3):
        if n_rays is None:
            n_rays = self.n_rays
        self.set_mcmc_prms(theta)
        logL_rays = 0.0
        for cnt, img in enumerate(self.fit_imgs):
            logL_rays += img.horizon_ray_logL(self.dem, n_rays=n_rays, eps=eps)
        logL_ant = 0
        for img in self.imgs:
            # not every image used for horizon/tie-point constraints
            # necessarily has a hand-picked antenna pixel
            if hasattr(img, 'meta') and 'ant_px' in img.meta:
                logL_ant += img.ant_logL(self.ant_pos, self.box_size)
        logL = logL_rays + logL_ant + self.tie_logL() + self.gps_delta_logL()
        return logL

    def export_jax(self, n_rays=None, eps=1e-3, dtype=dtype_r):
        if n_rays is None:
            n_rays = self.n_rays
        dem_pack = self.dem.export_jax(dtype=dtype)
        fit_statics = [img.export_jax(n_rays=n_rays, eps=eps, dtype=dtype)
                       for img in self.fit_imgs]
        all_statics = [img.export_jax(n_rays=n_rays, eps=eps, dtype=dtype)
                       for img in self.imgs]
        return dict(
            dem=dem_pack,
            fit=fit_statics,
            all=all_statics,
            ant_pos_prior=np.asarray(self.ant_pos_prior, dtype=dtype),
            box_size=dtype(self.box_size),
            sigmas=np.asarray(self.sigmas, dtype=dtype),
        )

