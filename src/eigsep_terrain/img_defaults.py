"""Load per-image defaults (ant_px, starting prms_u, img glob) from a single
JSON file so fit_image.py / plot_image_fit.py / tune_image.py stay in sync.
Edit defaults.json, not the scripts, when a starting value changes.
"""
import json
import os

DEFAULT_DEFAULTS_PATH = '/Users/komalkaur/Desktop/eigsep_stuff/eigsep_terrain/eigsep_terrain/defaults.json'


def load_defaults(path='/Users/komalkaur/Desktop/eigsep_stuff/eigsep_terrain/eigsep_terrain/defaults.json'):
    """Returns (img_glob, cache_file, DEFAULT_META, DEFAULT_PRMS_U_BY_KEY, IMG_KEYS)."""
    path = path or DEFAULT_DEFAULTS_PATH
    with open(path) as f:
        data = json.load(f)

    img_glob = data["img_glob"]
    cache_file = data.get("cache_file", "marjum_dem.npz")

    img_keys = list(data["images"].keys())
    default_meta = {k: {"ant_px": tuple(v["ant_px"])} for k, v in data["images"].items()}
    default_prms_u = {k: tuple(v["prms_u"]) for k, v in data["images"].items()}

    return img_glob, cache_file, default_meta, default_prms_u, img_keys