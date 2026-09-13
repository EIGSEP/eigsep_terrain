"""Terrain reflectivity at RF frequencies for EIGSEP.

Functions to compute the complex reflection coefficient (normal incidence)
from material properties, and a table of representative terrain types
for use at 50--250 MHz.

Workflow
--------
1. Choose a terrain type from ``TERRAIN_TYPES`` (or supply your own
   ``eps_r`` and ``resistivity_ohm_m``).
2. Convert resistivity to CGS conductivity with
   ``conductivity_from_resistivity``.
3. Get the complex refractive index with ``complex_ref_index``.
4. Get the complex (voltage) reflection coefficient with
   ``reflection_coefficient``; square the absolute value for power.

Alternatively, use ``terrain_reflection_coefficient`` as a one-shot
convenience wrapper.

GPR
---
If you have GPR data, ``eps_r_from_gpr`` inverts two-way travel time at a
known reflector depth to give the in-situ bulk ``eps_r``.
"""
import warnings
import numpy as np
from scipy.constants import c as C

C = np.float32(C)

# ---------------------------------------------------------------------------
# Terrain type table
# ---------------------------------------------------------------------------
# Representative (eps_r, resistivity [Ohm m]) values for dry materials at
# 50--250 MHz.  eps_r is the real relative permittivity; resistivity is the
# DC/low-frequency value (conductivity is frequency-independent at these
# frequencies for most dry rock).  Ranges are given in the description.
# Resistivity entries use the geometric mid-point of the published range.
TERRAIN_TYPES = {
    "granite": {
        "eps_r": 5.0,
        "resistivity_ohm_m": 1e4,
        "description": (
            "Dry granite or basalt. "
            "eps_r 4--6, resistivity 1e3--1e6 Ohm m."
        ),
    },
    "limestone": {
        "eps_r": 7.5,
        "resistivity_ohm_m": 3.2e3,
        "description": (
            "Limestone or dolomite. "
            "eps_r 6--9, resistivity 1e2--1e5 Ohm m."
        ),
    },
    "sandstone": {
        "eps_r": 5.5,
        "resistivity_ohm_m": 1e3,
        "description": (
            "Dry sandstone. "
            "eps_r 4--7, resistivity 1e2--1e4 Ohm m."
        ),
    },
    "dry_sand": {
        "eps_r": 4.0,
        "resistivity_ohm_m": 1e4,
        "description": (
            "Dry sand or gravel. "
            "eps_r 3--5, resistivity 1e3--1e6 Ohm m."
        ),
    },
    "wet_soil": {
        "eps_r": 22.0,
        "resistivity_ohm_m": 1e2,
        "description": (
            "Wet soil.  Water content dominates eps_r. "
            "eps_r 15--30, resistivity 10--1e3 Ohm m."
        ),
    },
    "lunar_regolith": {
        "eps_r": 3.0,
        "resistivity_ohm_m": 1e9,
        "description": (
            "Lunar regolith (dry, low-loss). "
            "eps_r 2.7--3.2 (Olhoeft & Strangway 1975), "
            "resistivity >1e8 Ohm m."
        ),
    },
}


# ---------------------------------------------------------------------------
# Low-level functions
# ---------------------------------------------------------------------------

def conductivity_from_resistivity(resistivity_ohm_m):
    """Return conductivity in CGS units [1/s] from resistivity [Ohm m]."""
    return 1 / (resistivity_ohm_m * 1.113e-12 * 100)


def complex_permittivity(eps_r, sigma, freqs):
    """Calculate the complex permittivity of a material.

    Parameters
    ----------
    eps_r : float
        Real relative permittivity.
    sigma : float
        Conductivity in CGS units [1/s].
        Use ``conductivity_from_resistivity`` to convert from [Ohm m].
    freqs : array_like
        Frequencies [Hz].

    Returns
    -------
    eps_tilde : numpy.ndarray
        Complex permittivity.
    """
    omega = 2 * np.pi * freqs
    return eps_r - 1j * (4 * np.pi * sigma) / omega


def complex_ref_index(eps_r, sigma, freqs):
    """Calculate the complex refractive index of a material.

    Parameters
    ----------
    eps_r : float
        Real relative permittivity.
    sigma : float
        Conductivity in CGS units [1/s].
    freqs : array_like
        Frequencies [Hz].

    Returns
    -------
    n_tilde : numpy.ndarray
        Complex refractive index.
    """
    return np.sqrt(complex_permittivity(eps_r, sigma, freqs))


def reflection_coefficient(eta, eta0=1):
    """Complex (voltage/field) reflection coefficient at normal incidence.

    Parameters
    ----------
    eta : complex array_like
        Refractive index of the second medium.
    eta0 : float, optional
        Refractive index of the first medium (default 1 for air/vacuum).

    Returns
    -------
    r : complex numpy.ndarray
        Field reflection coefficient.  For power use ``np.abs(r)**2``.
    """
    return (eta0 - eta) / (eta0 + eta)


# ---------------------------------------------------------------------------
# GPR helper
# ---------------------------------------------------------------------------

def eps_r_from_gpr(depth_m, travel_time_s):
    """Infer bulk relative permittivity from a GPR reflection.

    Given a reflector at known depth and the measured two-way travel time,
    returns eps_r = (c / v)^2 where v = 2*depth / travel_time.

    Parameters
    ----------
    depth_m : float
        Depth to the reflector [m].
    travel_time_s : float
        Two-way travel time to the reflector [s].

    Returns
    -------
    eps_r : float
        Bulk real relative permittivity of the material above the reflector.
    """
    v = 2 * depth_m / travel_time_s
    return float((C / v) ** 2)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def terrain_reflection_coefficient(terrain_type, freqs):
    """Reflection coefficient for a named terrain type.

    Parameters
    ----------
    terrain_type : str
        Key in ``TERRAIN_TYPES`` (e.g. ``'granite'``, ``'lunar_regolith'``).
    freqs : array_like
        Frequencies [Hz].

    Returns
    -------
    r : complex numpy.ndarray
        Field reflection coefficient at each frequency.
        For power use ``np.abs(r)**2``.
    """
    if terrain_type not in TERRAIN_TYPES:
        raise ValueError(
            f"Unknown terrain type {terrain_type!r}. "
            f"Available: {list(TERRAIN_TYPES)}"
        )
    t = TERRAIN_TYPES[terrain_type]
    sigma = conductivity_from_resistivity(t["resistivity_ohm_m"])
    n = complex_ref_index(t["eps_r"], sigma, freqs)
    return reflection_coefficient(n)


# ---------------------------------------------------------------------------
# Deprecated shim kept for backward compatibility
# ---------------------------------------------------------------------------

def permittivity_from_conductivity(conductivity, freqs):
    """Deprecated.  Use ``complex_ref_index(eps_r=1, sigma=..., freqs=...)``."""
    warnings.warn(
        "permittivity_from_conductivity is deprecated due to confusing "
        "naming. It actually returns refractive index, assuming relative "
        "permittivity of 1. Use complex_ref_index instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return complex_ref_index(1, conductivity, freqs)
