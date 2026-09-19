'''JAX ports of the HorizonImage likelihood terms in img.py.

These mirror ``pixels_to_rays``, ``HorizonImage.get_rays``,
``HorizonImage.horizon_ray_logL`` and ``HorizonImage.ant_logL`` so that a
full-problem logL can be jitted and vmapped (see solver_jax.py).
'''
import jax
import jax.numpy as jnp
from .ray_jax import ray_distance_coarse_to_fine

dtype_r = jnp.float32


def rot_m_jax(angle, axis, dtype=dtype_r):
    '''Right-handed rotation matrix about unit vector ``axis`` by
    ``angle``; JAX equivalent of utils.rot_m for a single axis.'''
    x, y, z = axis
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    C = 1.0 - c
    return jnp.array([
        [c + x * x * C,     x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C,     y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ], dtype=dtype)


def pixels_to_rays_jax(Nu, Nv, f, uv, dtype=dtype_r):
    '''JAX equivalent of img.pixels_to_rays for explicit (u, v) pixels.
    Returns unit rays of shape (3, *u.shape).'''
    u, v = uv
    u = jnp.asarray(u, dtype=dtype)
    v = jnp.asarray(v, dtype=dtype)
    rays = jnp.stack([
        (Nu // 2).astype(dtype) - u,
        (Nv // 2).astype(dtype) - v,
        jnp.full(u.shape, f, dtype=dtype),
    ], axis=0)
    return rays / jnp.linalg.norm(rays, axis=0, keepdims=True)


@jax.jit
def get_rays_jax(Nu, Nv, f, th, ph, ti, u_px, v_px):
    '''JAX equivalent of HorizonImage.get_rays.'''
    z_rays = pixels_to_rays_jax(Nu, Nv, f, (u_px, v_px))
    rm_tilt = rot_m_jax(ti, (0.0, 0.0, 1.0))
    rm_th = rot_m_jax(th, (0.0, 1.0, 0.0))
    rm_ph = rot_m_jax(ph, (0.0, 0.0, 1.0))
    rm = rm_ph @ (rm_th @ rm_tilt)
    return jnp.einsum('ij,j...->i...', rm, z_rays)


def _horizon_ray_logL_jax(E, N, U, Nu, Nv,
                          e, n, u, th, ph, ti, f,
                          x_px, y_px, psky,
                          eps=1e-3, correlation_px=None,
                          max_iters=(4096, 4096)):
    '''JAX equivalent of HorizonImage.horizon_ray_logL. If correlation_px
    is None, per-pixel terms are summed as independent; otherwise the mean
    is rescaled by the same effective sample count as the NumPy version.'''
    rays = get_rays_jax(Nu, Nv, f, th, ph, ti, x_px, y_px)
    start_point = jnp.array([e, n, u], dtype=dtype_r)
    r = ray_distance_coarse_to_fine(E, N, U, start_point, rays,
                                    delta_steps=(5.0, 1.0),
                                    max_iters=max_iters)
    model_sky = jnp.isnan(r)
    p = jnp.clip(psky, eps, 1.0 - eps)
    per_pixel = jnp.where(model_sky, jnp.log(p), jnp.log1p(-p))
    if correlation_px is None:
        return jnp.sum(per_pixel)
    n_rays = per_pixel.shape[0]
    col_span = jnp.maximum(jnp.max(y_px) - jnp.min(y_px), 1)
    n_eff = jnp.clip(col_span / correlation_px, 1, n_rays)
    return jnp.mean(per_pixel) * n_eff


def _ant_logL_jax(Nu, Nv,
                  e, n, u, th, ph, ti, f,
                  ant_e, ant_n, ant_u,
                  ant_u_px, ant_v_px,
                  box_size):
    '''JAX equivalent of HorizonImage.ant_logL.'''
    ant_ray = get_rays_jax(Nu, Nv, f, th, ph, ti, ant_u_px, ant_v_px)
    ant_ray = ant_ray.reshape(3,)
    r_ant = jnp.array([ant_e - e, ant_n - n, ant_u - u], dtype=dtype_r)
    r_norm = jnp.linalg.norm(r_ant)
    # atan2 form of the angle between rays: same value as the NumPy
    # arccos(cos) but with finite gradients near delta_theta = 0
    delta_theta = jnp.arctan2(jnp.linalg.norm(jnp.cross(ant_ray, r_ant)),
                              jnp.dot(ant_ray, r_ant))
    sigma_theta = box_size / r_norm
    return -0.5 * jnp.log(2.0 * jnp.pi * sigma_theta**2) \
        - 0.5 * (delta_theta / sigma_theta)**2


# max_iters is a tuple, so it must be static rather than traced
horizon_ray_logL_jax = jax.jit(_horizon_ray_logL_jax,
                               static_argnames=('max_iters',))
ant_logL_jax = jax.jit(_ant_logL_jax)
