'''Save/load a "fit": a set of per-image camera poses plus an antenna
position, as one self-describing .npz file. This is the interface between
whatever produced a fit (Phase 2's per-image refine_pose, the bundle-
adjustment loop, or an MCMC trace's posterior mean) and anything that wants
to inspect one -- so an inspection notebook can work on any of them without
caring how they were made.

File contents:
    keys        : (N,) array of image key strings, e.g. '2223'
    <key>_prms  : (7,) array in PRM_ORDER for that image, for each key
    ant_pos     : (3,) array, antenna position in ENU meters
    ant_pos_std : (3,) array, optional -- posterior std if the fit came
                  from an MCMC trace; omitted for deterministic fits
'''

import numpy as np
from .img import PRM_ORDER


def save_fit(path, prms_by_key, ant_pos, ant_pos_std=None):
    '''prms_by_key: dict of image key -> dict with all PRM_ORDER keys (or
    -> a HorizonImage, whose .prms is used). ant_pos: (3,) ENU meters.
    ant_pos_std: optional (3,) array (e.g. MCMC posterior std).'''
    keys = list(prms_by_key.keys())
    arrs = {}
    for k in keys:
        v = prms_by_key[k]
        prms = v.prms if hasattr(v, 'prms') else v
        arrs[f'{k}_prms'] = np.array([prms[p] for p in PRM_ORDER], dtype=float)
    arrs['keys'] = np.array(keys)
    arrs['ant_pos'] = np.asarray(ant_pos, dtype=float)
    if ant_pos_std is not None:
        arrs['ant_pos_std'] = np.asarray(ant_pos_std, dtype=float)
    np.savez(path, **arrs)


def load_fit(path):
    '''Returns (prms_by_key, ant_pos, ant_pos_std). prms_by_key is a dict
    of image key -> dict with all PRM_ORDER keys. ant_pos_std is None if
    the fit didn't record one.'''
    npz = np.load(path)
    keys = [str(k) for k in npz['keys']]
    prms_by_key = {k: dict(zip(PRM_ORDER, npz[f'{k}_prms'])) for k in keys}
    ant_pos = npz['ant_pos']
    ant_pos_std = npz['ant_pos_std'] if 'ant_pos_std' in npz.files else None
    return prms_by_key, ant_pos, ant_pos_std


def fit_from_trace(trace, keys, dem):
    '''Build a (prms_by_key, ant_pos, ant_pos_std) triple from an arviz/
    pymc MCMC trace (e.g. one of Phase 6's phase6_stepN_trace.nc), taking
    posterior means (and, for the antenna, stds) -- so an MCMC result can
    be saved via save_fit and inspected with the same tools as a
    deterministic bundle-adjustment fit.'''
    prms_by_key = {}
    for key in keys:
        d = {}
        for k in PRM_ORDER:
            if k == 'u':
                e = np.asarray(trace.posterior[f'{key}_e']).flatten().mean()
                n = np.asarray(trace.posterior[f'{key}_n']).flatten().mean()
                logh = np.asarray(trace.posterior[f'{key}_log_h']).flatten().mean()
                u0 = float(dem.interp_alt(np.array([e]), np.array([n]))[0])
                d[k] = np.exp(logh) + u0
            else:
                d[k] = np.asarray(trace.posterior[f'{key}_{k}']).flatten().mean()
        prms_by_key[key] = d

    ant_e = np.asarray(trace.posterior['ant_e']).flatten()
    ant_n = np.asarray(trace.posterior['ant_n']).flatten()
    ant_logh = np.asarray(trace.posterior['ant_log_h']).flatten()
    ant_u0 = float(dem.interp_alt(np.array([ant_e.mean()]), np.array([ant_n.mean()]))[0])
    ant_u = np.exp(ant_logh) + ant_u0
    ant_pos = np.array([ant_e.mean(), ant_n.mean(), ant_u.mean()])
    ant_pos_std = np.array([ant_e.std(), ant_n.std(), ant_u.std()])
    return prms_by_key, ant_pos, ant_pos_std
