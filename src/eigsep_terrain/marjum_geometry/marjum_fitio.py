"""Fit-file input/output for Marjum photogrammetry products.

Restores the ``load_fit``/``save_fit`` interface that ``eigsep_terrain.fitio``
used to provide.  That module no longer exists in the installed package, which
leaves ``marjum_bundle``, ``marjum_mcmc``, ``marjum_guided``, ``marjum_cv_*``
unimportable.  The format is recovered from the saved products themselves, so
existing ``.npz`` files load unchanged; nothing here is new science.

Two on-disk layouts occur and both are read:

``prms``
    One ``{key}_prms`` array of seven camera parameters per image, a ``keys``
    array, and ``ant_pos``.  Written by ``save_fit``.  No distortion.
``state``
    Stacked ``cameras`` ``(n, 7)`` with ``keys``, ``antenna``, and usually
    ``distortion``/``groups``/``points``/``obs_*``.  Written by the bundle and
    add-view pipelines.

``load_fit`` returns ``(poses, antenna, extra)`` where ``poses`` maps key to a
dict over :data:`marjum_bundle.PRM_ORDER` and ``extra`` carries every remaining
array, so distortion and track geometry survive a round trip through callers
that only need the poses.
"""
from pathlib import Path

import numpy as np

from marjum_bundle import PRM_ORDER


def load_fit(path):
    """Read a fit file as ``(poses, antenna, extra)``."""
    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        names = set(data.files)
        if 'keys' not in names:
            raise ValueError(f'{path}: not a fit file, no "keys" array')
        keys = [str(k) for k in data['keys']]
        if 'cameras' in names:
            cameras = np.asarray(data['cameras'], float)
            if cameras.shape != (len(keys), len(PRM_ORDER)):
                raise ValueError(f'{path}: cameras {cameras.shape} does not match '
                                 f'{len(keys)} keys x {len(PRM_ORDER)} parameters')
            consumed = {'keys', 'cameras'}
        else:
            missing = [k for k in keys if f'{k}_prms' not in names]
            if missing:
                raise ValueError(f'{path}: missing camera parameters for {missing}')
            cameras = np.array([np.asarray(data[f'{k}_prms'], float) for k in keys])
            consumed = {'keys'} | {f'{k}_prms' for k in keys}
        antenna_name = 'antenna' if 'antenna' in names else 'ant_pos'
        if antenna_name not in names:
            raise ValueError(f'{path}: no antenna position')
        antenna = np.asarray(data[antenna_name], float).reshape(3)
        consumed.add(antenna_name)
        poses = {k: dict(zip(PRM_ORDER, c)) for k, c in zip(keys, cameras)}
        extra = {n: data[n] for n in data.files if n not in consumed}
    return poses, antenna, extra


def save_fit(path, poses, antenna, **extra):
    """Write poses and antenna in the ``prms`` layout used by ``load_fit``."""
    keys = list(poses)
    arrays = {f'{k}_prms': np.array([float(poses[k][q]) for q in PRM_ORDER])
              for k in keys}
    arrays['keys'] = np.array(keys)
    arrays['ant_pos'] = np.asarray(antenna, float).reshape(3)
    for name, value in extra.items():
        if name in arrays:
            raise ValueError(f'extra array {name!r} collides with the fit layout')
        arrays[name] = value
    np.savez(Path(path), **arrays)
