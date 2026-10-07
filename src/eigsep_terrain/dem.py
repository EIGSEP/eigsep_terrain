'''Tools for dealing with digital elevation models.'''

import numpy as np
import PIL.Image
import os
import pyuvdata
import xmltodict
from pyproj import CRS, Proj, Transformer
from .utils import az_bin, calc_az_bin_range, calc_rmin
from .ray import ray_trace_basic, healpix_rays, calc_maxiter

dtype_r = np.float32

XML_CRD_KEYWORDS = ('eastbc', 'westbc', 'northbc', 'southbc')
DEFAULT_BACKEND = 'numpy'
CACHE_VERSION = 2

class DEM(dict):
    '''Class for interacting with Digital Elevation Model data.'''
    
    def __init__(self, cache_file=None, clear_cache=False, backend=DEFAULT_BACKEND):
        self._cache_file = cache_file
        self.backend = backend
        self.e0_px = 0
        self.n0_px = 0
        if clear_cache and cache_file is not None and os.path.exists(cache_file):
            # A pre-UTM cache may be a pinned data product. Never destroy it
            # while changing coordinate conventions.
            with np.load(cache_file) as npz:
                self._require_current_cache(npz)
            os.remove(cache_file)
        if cache_file is not None and os.path.exists(cache_file):
            self.load_cache()

    def load_cache(self):
        '''Retrieve cached DEM data from npz file.'''
        with np.load(self._cache_file) as npz:
            self._require_current_cache(npz)
            self._set_projection(int(npz['crs_epsg']))
            self.raster_origin = npz['raster_origin']
            self.files = npz['files']
            self.res = npz['res']
            self.data = npz['dem']
            self.map_crd = {k: npz[k] for k in XML_CRD_KEYWORDS}
            self.survey_offset = npz['survey_offset']
            self.e0_px = int(npz['e0_px'])
            self.n0_px = int(npz['n0_px'])

    def _require_current_cache(self, npz):
        if ('cache_version' not in npz
                or int(npz['cache_version']) != CACHE_VERSION):
            raise ValueError(
                f'{self._cache_file} uses an incompatible DEM coordinate frame; '
                'preserve it and build a new cache path from its exact source '
                'GeoTIFF tile list.'
            )

    def save_cache(self):
        '''Cache DEM data in npz file.'''
        if self._cache_file is not None:
            np.savez(self._cache_file, cache_version=CACHE_VERSION,
                     crs_epsg=self.crs.to_epsg(),
                     raster_origin=self.raster_origin,
                     dem=self.data, res=self.res,
                     files=self.files, survey_offset=self.survey_offset,
                     e0_px=self.e0_px, n0_px=self.n0_px,
                     **self.map_crd)

    def _set_projection(self, epsg):
        self.crs = CRS.from_epsg(epsg)
        if not self.crs.is_projected or self.crs.utm_zone is None:
            raise ValueError('DEM requires a UTM projected CRS')
        self._forward = Transformer.from_crs(
            self.crs.geodetic_crs, self.crs, always_xy=True)
        self._inverse = Transformer.from_crs(
            self.crs, self.crs.geodetic_crs, always_xy=True)

    def load_tif(self, files, survey_offset=(0, 0, 0)):
        """Load a rectangular UTM mosaic (files indexed east, then north).

        Coordinates refer to pixel centres. All tiles must share a CRS,
        resolution, shape and contiguous north-up georeferencing.
        """
        files = np.asarray(files)
        tiles = {}
        for i, j in np.ndindex(files.shape):
            with PIL.Image.open(files[i, j]) as image:
                tags = image.tag_v2
                tie = tags[33922]
                sx, sy, _ = tags[33550]
                keys = tags[34735]
                geo = {keys[k]: tuple(keys[k+1:k+4])
                       for k in range(4, len(keys), 4)}
                epsg = geo[3072][2]
                raster_type = geo.get(1025, (0, 1, 1))[2]
                if 34264 in tags or sx <= 0 or sx != sy:
                    raise ValueError('Expected square, north-up pixels')
                if raster_type not in (1, 2):
                    raise ValueError('Unsupported GeoTIFF raster type')
                width, height = image.size
                x = tie[3] - tie[0] * sx
                y = tie[4] + tie[1] * sy
                # PixelIsArea tiepoints describe edges; PixelIsPoint centres.
                half = 0.5 if raster_type == 1 else 0.0
                origin = np.array([x + half*sx,
                                   y - (height-1+half)*sy])
                if i == j == 0:
                    self._set_projection(epsg)
                    self.res = sx
                    self.raster_origin = origin
                    shape = (height, width)
                expected = self.raster_origin + np.array(
                    [i * width * sx, j * height * sy])
                if (epsg != self.crs.to_epsg() or sx != self.res
                        or (height, width) != shape
                        or not np.allclose(origin, expected, rtol=0,
                                           atol=1e-6)):
                    raise ValueError('Tiles do not form a contiguous mosaic')
                tiles[i, j] = np.flipud(np.array(image, dtype='float32'))
        self.data = np.hstack([np.vstack([tiles[i, j]
                              for j in range(files.shape[1])])
                              for i in range(files.shape[0])])
        self.files = files
        self.e0_px = self.n0_px = 0
        self.survey_offset = np.asarray(survey_offset, dtype=float)

    def latlon_to_raster(self, lat, lon, alt=None, survey_offset=None):
        """Geographic degrees to local UTM grid metres, with unchanged height.

        Lat/lon use the raster's geographic datum (NAD83(2011) at Marjum).
        WGS84 GPS coordinates need a separate datum correction (~1 m here).
        Heights must already use the DEM vertical datum; no geoid conversion
        is performed. The survey offset is subtracted from all three axes.
        """
        offset = self.survey_offset if survey_offset is None else survey_offset
        e, n = self._forward.transform(float(lon), float(lat))
        return np.array([e-self.raster_origin[0],
                         n-self.raster_origin[1],
                         0 if alt is None else float(alt)]) - offset

    def raster_to_latlon(self, enu, survey_offset=None):
        """Inverse of latlon_to_raster, returning degrees/degrees/metres."""
        offset = self.survey_offset if survey_offset is None else survey_offset
        e, n, u = np.asarray(enu) + offset
        lon, lat = self._inverse.transform(
            e+self.raster_origin[0], n+self.raster_origin[1])
        return lat, lon, u

    def grid_convergence(self, e, n):
        """True minus grid azimuth in radians at a raster position."""
        lat, lon, _ = self.raster_to_latlon([e, n, 0])
        return np.deg2rad(Proj(self.crs).get_factors(
            lon, lat).meridian_convergence)

    def _azimuth_offset(self, e, n, frame):
        if frame == 'grid':
            return 0.0
        if frame != 'true':
            raise ValueError("azimuth_frame must be 'true' or 'grid'")
        # Synthetic grids without georeferencing retain their local axes.
        return self.grid_convergence(e, n) if hasattr(self, 'crs') else 0.0

    def load_xml(self, filename):
        with open(filename, 'rb') as f:
           self.map_crd = {k: np.deg2rad(float(v)) for k, v in
     xmltodict.parse(f)['metadata']['idinfo']['spdom']['bounding'].items()}

    def latlon_to_enu(self, lat, lon, alt=None, survey_offset=None,
                      frame="utm"):
        """Convert degrees/degrees/metres to UTM raster coordinates.

        frame='tangent' reproduces the historical tangent-plane conversion.
        See latlon_to_raster for datum and height conventions.
        """
        if frame == "utm":
            return self.latlon_to_raster(lat, lon, alt, survey_offset)
        if frame != "tangent":
            raise ValueError("frame must be utm or tangent")
        lat = np.deg2rad(float(lat))
        lon = np.deg2rad(float(lon))
        if alt is None:
            alt = 0
        else:
            alt = float(alt)
        if survey_offset is None:
            survey_offset = self.survey_offset
        ecef = pyuvdata.utils.XYZ_from_LatLonAlt(lat, lon, 0)
        # XXX don't understand negative alt below
        enu = pyuvdata.utils.ENU_from_ECEF(ecef,
                latitude=self.map_crd['southbc'],
                longitude=self.map_crd['westbc'], altitude=-alt)
        return enu - survey_offset

    def enu_to_latlon(self, enu, survey_offset=None, frame="utm"):
        """Inverse grid conversion; frame='tangent' uses the legacy frame."""
        if frame == "utm":
            return self.raster_to_latlon(enu, survey_offset)
        if frame != "tangent":
            raise ValueError("frame must be utm or tangent")
        alt = 0
        if survey_offset is None:
            survey_offset = self.survey_offset
        xyz = pyuvdata.utils.ECEF_from_ENU(
                enu + survey_offset,
                latitude=self.map_crd['southbc'],
                longitude=self.map_crd['westbc'],
                altitude=-alt
        )
        lat, lon, alt = pyuvdata.utils.LatLonAlt_from_XYZ(xyz)
        return np.rad2deg(lat), np.rad2deg(lon), alt

    def m2px(self, *args, axis=None, res=None):
        '''Convert meters to array-index pixels: round(m / res) + offset,
        where offset is e0_px/n0_px for axis='e'/'n' (0 for axis=None, the
        legacy no-offset behavior). The offset is what lets the array grow
        in a given direction (e.g. a newly downloaded tile to the south)
        without changing what any already-recorded e_m/n_m value resolves
        to -- see load_cache's e0_px/n0_px docstring note.'''
        if res is None:
            res = self.res
        offset = {'e': self.e0_px, 'n': self.n0_px}.get(axis, 0)
        px = tuple(np.around(m / res).astype(int) + offset for m in args)
        return px

    def interp_alt(self, e_m, n_m, return_vec=False):
        (e_px,) = self.m2px(e_m, axis='e')
        (n_px,) = self.m2px(n_m, axis='n')
        u_m = self.data[n_px, e_px]
        if return_vec:
            try:
                return np.concatenate([e_m, n_m, u_m], axis=0)
            except(ValueError):
                return np.array([e_m, n_m, u_m])
        else:
            return u_m

    def add_survey_points(self, pnts, survey_offset=None):
        self.update({k: self.latlon_to_enu(*v.split(', '), survey_offset=survey_offset) for k, v in pnts.items()})

    def get_en(self, erng_m=None, nrng_m=None, return_px=False,
                     decimate=1, edges=False):
        if erng_m is None:
            emn, emx = 0, self.data.shape[1]
        else:
            emn, emx = self.m2px(*erng_m, axis='e')
        if nrng_m is None:
            nmn, nmx = 0, self.data.shape[0]
        else:
            nmn, nmx = self.m2px(*nrng_m, axis='n')
        if edges:
            _E = np.arange(emn, emx + decimate, decimate) - 0.5
            _N = np.arange(nmn, nmx + decimate, decimate) - 0.5
        else:
            _E = np.arange(emn, emx, decimate)
            _N = np.arange(nmn, nmx, decimate)
        if return_px:
            return _E, _N
        else:
            return (_E - self.e0_px) * self.res, (_N - self.n0_px) * self.res

    def get_tile(self, erng_m=None, nrng_m=None, mesh=True, decimate=1):
        _E, _N = self.get_en(erng_m, nrng_m, return_px=True, decimate=decimate)
        U = self.data[_N][:, _E]
        if mesh:
            E, N = np.meshgrid(_E, _N)
        else:
            E, N = _E, _N
        return (E - self.e0_px) * self.res, (N - self.n0_px) * self.res, U

    def export_jax(self, dtype=dtype_r):
        (E, N) = self.get_en()
        U = self.data
        return dict(
            E=np.asarray(E, dtype=dtype),
            N=np.asarray(N, dtype=dtype),
            U=np.asarray(U, dtype=dtype),
        )


    def zone_of_avoidance_height(self, e_m, n_m, r_zoa=100, decimate=1):
        '''Return the height needed to enforce all terrain is
        a distance > r_zoa away.'''
        E, N, U = self.get_tile(mesh=False, decimate=decimate)
        res = self.res * decimate
        k = np.around(r_zoa / res).astype(int)
        dr = np.arange(-k, k+1) * res
        rs2 = dr[:, None]**2 + dr[None, :]**2
        root = np.sqrt((r_zoa**2 - rs2).clip(0))
        (e_px,) = self.m2px(e_m, axis='e')
        (n_px,) = self.m2px(n_m, axis='n')
        h = np.zeros_like(e_m)
        for i in range(e_px.size):
            ei, ni = e_px[i], n_px[i]
            if ni - k < 0 or ei - k < 0:
                continue
            if ni + k + 1 > U.shape[0] or ei + k + 1 > U.shape[1]:
                continue
            h[i] = np.max(U[ni-k:ni+k+1, ei-k:ei+k+1] + root) - U[ni, ei]
        return h

    def find_anchors(self, e0, n0, u0, decimate=1, boundary=False,
                     n_anchors=2, r_anchor_max=300,
                     min_angle=np.deg2rad(20), n_az_bins=240):
        '''Find opposing anchor positions.
        r_anchor_max: meters, min_angle: radians.'''
        # Find anchor points
        erng = (e0 - r_anchor_max, e0 + r_anchor_max)
        nrng = (n0 - r_anchor_max, n0 + r_anchor_max)
        E, N, U = self.get_tile(erng, nrng, mesh=False, decimate=decimate)
        rdist = np.sqrt((E[None, :] - e0)**2 + (N[:, None] - n0)**2)
        cone = u0 + np.tan(min_angle) * rdist
        inds = (U - cone > 0)
        if not np.any(inds):
            return [], [] if boundary else []
        rmin = r_anchor_max * np.ones(n_az_bins)
        rmax = np.zeros(n_az_bins)
        b = az_bin(E - e0, N - n0, n_az_bins)
        for _r, _b in zip(rdist[inds], b[inds]):
            rmin[_b % n_az_bins] = min(rmin[_b % n_az_bins], _r)
            rmax[_b % n_az_bins] = max(rmax[_b % n_az_bins], _r)
        # Assign inf to areas that don't meet anchor length requirements
        rmin = np.where(rmin >= r_anchor_max, np.inf, rmin)
        # Fold to enforce anchors being on opposite sides,
        # then minimize total anchor length
        rmin.shape = (n_anchors, -1)
        rmax.shape = (n_anchors, -1)
        rtot = np.sum(rmin, axis=0)
        bmin = np.argmin(rtot)
        r_anchors = rmin[:, bmin]
        az_min = bmin * 2 * np.pi / n_az_bins
        az_anchors = az_min + 2 * np.pi / n_anchors * np.arange(n_anchors)
        anchors_e = e0 + r_anchors * np.sin(az_anchors)
        anchors_n = n0 + r_anchors * np.cos(az_anchors)
        if not boundary:
            return list(zip(anchors_e, anchors_n))
        # find boundary
        valid = np.where(rtot < n_anchors * r_anchor_max)[0]
        az_valid = valid * 2 * np.pi / n_az_bins
        boundary = []
        for a in range(n_anchors):
            bound_min = [(e0 + r * np.sin(az), n0 + r * np.cos(az))
                         for r, az in zip(rmin[a, valid],
                                     az_valid + a * 2 * np.pi / n_anchors)]
            bound_max = [(e0 + r * np.sin(az), n0 + r * np.cos(az))
                         for r, az in zip(rmax[a, valid],
                                     az_valid + a * 2 * np.pi / n_anchors)]
            boundary.append(bound_min + bound_max[::-1])
        return list(zip(anchors_e, anchors_n)), np.array(boundary)

    def build_maxpool_pyramid(self, data=None, factor=4):
        '''Return a list of (2D array, factor) pairs, each downsampled
        along 2 dimensions by the specified factor and maxpooled.'''
        if data is None:
            data = self.data
        answer = [(data, 1)]
        if max(data.shape) <= factor:
            return answer
        # Preserve partial boundary blocks; zero padding loses tall terrain.
        pad = tuple((0, (-size) % factor) for size in data.shape)
        padded = np.pad(np.asarray(data, dtype=np.result_type(data.dtype, np.float32)), pad,
                        constant_values=-np.inf)
        blocks = padded.reshape(padded.shape[0] // factor, factor,
                                padded.shape[1] // factor, factor)
        pool_data = np.max(blocks, axis=(1, 3))
        answer += [(d, factor * f) for (d, f) in self.build_maxpool_pyramid(data=pool_data, factor=factor)]
        return answer

    def calc_horizon(self, e0, n0, u0, n_az=256, imp=None, f_prev=None,
                     ei_off=None, ni_off=None, crds=None, hangles=None,
                     azimuth_frame="true", _az_offset=None):
        """Return conservative bin maxima and their raster (north, east).

        Each bin is the maximum over pixels touching its angular interval,
        not a point sample. Use n_az >= 1440 for interpolated products.
        True azimuth uses observer convergence; grid preserves legacy axes.
        """
        if _az_offset is None:
            _az_offset = self._azimuth_offset(e0, n0, azimuth_frame)
        if imp is None:
            # top case
            imp = self.build_maxpool_pyramid()
            if hangles is None:
                hangles = np.zeros(n_az)
            crds = np.zeros([2, hangles.size], dtype=float)
            U, f = imp[-1]
            e_edges, n_edges = self.get_en(edges=True, decimate=f)
            _ni, _ei = 0, 0
        else:
            _U, f = imp[-1]
            f_step = (f_prev // f)
            _ni, _ei = ni_off * f_step, ei_off * f_step
            _e_edges, _n_edges = self.get_en(edges=True, decimate=f)
            n_edges = _n_edges[_ni:_ni + f_step + 1]
            e_edges = _e_edges[_ei:_ei + f_step + 1]
            U = _U[_ni:_ni + f_step, _ei:_ei + f_step]
    
        r_min = calc_rmin(e_edges, n_edges, e0, n0)
        az_min, az_max = calc_az_bin_range(e_edges, n_edges, e0, n0, n_az,
                          az_offset=_az_offset)
        hor_ang = np.arctan2(U - u0, r_min)
        # process in order of maximum possible horizon angle first
        n_pxs, e_pxs = np.unravel_index(np.argsort(-hor_ang, axis=None), r_min.shape)
        for cnt, (ni, ei) in enumerate(zip(n_pxs, e_pxs)):
            bmin = az_min[ni, ei]
            bmax = az_max[ni, ei]
            h = hor_ang[ni, ei]
            if bmin < bmax:
                slices = [slice(bmin, bmax)]
            elif bmin == bmax:
                slices = [slice(bmin, bmin+1)]
            else:
                slices = [slice(bmin, None), slice(0, bmax)]
            if len(imp) == 1:
                # base case
                for s in slices:
                    update = (hangles[s] < h)
                    crds[0,s] = np.where(update, self.res*(_ni+ni-self.n0_px), crds[0,s])
                    crds[1,s] = np.where(update, self.res*(_ei+ei-self.e0_px), crds[1,s])
                    # sets to highest value
                    hangles[s] = np.where(update, h, hangles[s])
            elif np.any(np.concatenate([hangles[s] < h for s in slices])):
                # need to recursively process at higher resolution
                hangles, crds = self.calc_horizon(e0, n0, u0,
                                        n_az=n_az, imp=imp[:-1], f_prev=f,
                                        ei_off=_ei+ei, ni_off=_ni+ni,
                                        crds=crds, hangles=hangles,
                                        azimuth_frame=azimuth_frame,
                                        _az_offset=_az_offset)
            else:
                # can skip this pixel
                pass
        return hangles, crds

    def ray_trace(self, start_point, nside, delta_r_m=1,
                  r_max=None, max_horizon_ang_deg=45, dtype=dtype_r,
                  backend=None, azimuth_frame="true"):
        '''Return the distance along a HealPix grid of specified nside from a
        ENU starting point until a ray intersects the terrain, in steps of
        delta_r_m [m], out to r_max (or map edge if None). Rays with elevation
        above max_horizon_ang_deg [deg] are assumed not to intersect terrain
        and are returned as NaN. Returns distance [m] in HealPix order, with
        non-intersecting rays set to NaN.

        azimuth_frame: 'true' (default) or 'grid'. True bearings are rotated
        into raster axes at the observer before tracing.
        backend: 'numpy', 'numba', or 'jax'. Defaults to self.backend.'''
        if backend is None:
            backend = self.backend
        E, N = self.get_en()
        rays = healpix_rays(nside, dtype=dtype)
        gamma = self._azimuth_offset(*start_point[:2], azimuth_frame)
        c, s = np.cos(gamma), np.sin(gamma)
        rays[:2] = np.array([[c, -s], [s, c]]) @ rays[:2]
        r_start = np.full(rays.shape[1], delta_r_m, dtype=dtype)
        if max_horizon_ang_deg is not None:
            above_horizon = rays[2] > np.sin(np.deg2rad(max_horizon_ang_deg))
            r_start[above_horizon] = np.nan
        else:
            above_horizon = np.zeros(rays.shape[1], dtype=bool)
        max_iter = calc_maxiter(E, N, self.data, start_point,
                                delta_r_m=delta_r_m, r_max=r_max)
        if backend == 'numpy':
            return ray_trace_basic(E, N, self.data, start_point, rays,
                                   delta_r_m=delta_r_m, r_start=r_start,
                                   max_iter=max_iter, dtype=dtype)
        elif backend == 'numba':
            from .ray_numba import ray_trace_basic_numba
            return ray_trace_basic_numba(E, N, self.data, start_point, rays,
                                         delta_r_m=delta_r_m, r_start=r_start,
                                         max_iter=max_iter, dtype=dtype)
        elif backend == 'jax':
            from .ray_jax import ray_trace_basic_jax_jit
            # Filter inactive rays before calling: JAX evaluates the body for
            # all Nr rays every step (lax.while_loop cannot prune dynamically),
            # so excluding above-horizon rays halves the per-step work.
            active_mask = ~above_horizon
            r_full = np.full(rays.shape[1], np.nan, dtype=dtype)
            if active_mask.any():
                r_sub = ray_trace_basic_jax_jit(
                    E, N, self.data, start_point, rays[:, active_mask],
                    delta_r_m=float(delta_r_m), max_iter=max_iter,
                )
                r_full[active_mask] = np.array(r_sub)
            return r_full
        else:
            raise ValueError(
                f"Unknown backend {backend!r}. "
                "Choose 'numpy', 'numba', or 'jax'."
            )
