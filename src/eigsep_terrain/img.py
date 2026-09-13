import os
import numpy as np
from matplotlib.image import imread
from healjax.coord import rot_m
from .utils import mask_near_horizon, fill_psky_holes
from .ray_numba import ray_distance_coarse_to_fine_numba
from .seg import TiledSkyProbSegFormer
from transformers import pipeline
import torch
import cv2
import pymc as pm

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
        self.key = os.path.basename(filename).split('_')[-1].split('.')[0]
        self.npzfile = 'img_seg_' + os.path.basename(filename).replace('jpg','npz')
        self.img = np.flipud(imread(self.filename))
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
        
    def horizon_ray_logL(self, dem, n_rays=1000, dtype=dtype_r, eps=1e-3):
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
        logL = np.sum(np.where(model_sky, logp_sky, logp_ground))
        return logL

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
            logL_ant += img.ant_logL(self.ant_pos, self.box_size)
        logL = logL_rays + logL_ant 
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

