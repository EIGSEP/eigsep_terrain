import jax
import jax.numpy as jnp
from jax import lax

dtype_r = jnp.float32
dtype_i = jnp.int32

def ray_trace_basic_jax(E, N, U, start_point, rays, delta_r_m=1.0,
                        r_start=None, max_iter=4096, dtype=dtype_r):
    E = jnp.asarray(E, dtype=dtype)
    N = jnp.asarray(N, dtype=dtype)
    U = jnp.asarray(U, dtype=dtype)
    sp = jnp.asarray(start_point, dtype=dtype)
    rays = jnp.asarray(rays, dtype=dtype)
    delta_r_m = jnp.asarray(delta_r_m, dtype=dtype)

    Ne = E.shape[0]
    Nn = N.shape[0]
    Nr = rays.shape[1]

    E0, Emax = E[0], E[-1]
    N0, Nmax = N[0], N[-1]

    dE = (E[1] - E[0])
    dN = (N[1] - N[0])
    inv_dE = 1.0 / dE
    inv_dN = 1.0 / dN

    u_max = jnp.max(U)

    if r_start is None:
        r0 = jnp.full((Nr,), 0, dtype=dtype)
        active0 = jnp.ones((Nr,), dtype=jnp.bool_)
    else:
        r0 = jnp.asarray(r_start, dtype=dtype)
        active0 = ~jnp.isnan(r0)

    def step_fn(state):
        i, r, active = state

        pts = sp[:, None] + r[None, :] * rays
        px, py, pz = pts[0], pts[1], pts[2]

        in_bounds = (px >= E0) & (px <= Emax) & (py >= N0) & (py <= Nmax)
        live = active & in_bounds
        oob = active & (~in_bounds)

        e_px = jnp.floor((px - E0) * inv_dE).astype(dtype_i)
        n_px = jnp.floor((py - N0) * inv_dN).astype(dtype_i)
        e_px_c = jnp.clip(e_px, 0, Ne - 1)
        n_px_c = jnp.clip(n_px, 0, Nn - 1)
        u_m = U[n_px_c, e_px_c]

        active_next = live & (pz > u_m)
        r_next = jnp.where(active_next, r + delta_r_m, r)
        # check out of bounds
        r_next = jnp.where(oob, jnp.nan, r_next)
        return (i + 1, r_next, active_next)

    def cond_fn(state):
        i, r, active = state
        return (i < max_iter) & jnp.any(active)

    _, r_final, active_final = lax.while_loop(cond_fn, step_fn, (dtype_i(0), r0, active0))
    r_final = jnp.where(active_final, jnp.nan, r_final)
    return r_final

## JIT-compiled callable (max_iter treated as static)
ray_trace_basic_jax_jit = jax.jit(ray_trace_basic_jax, static_argnames=("max_iter",))

def ray_distance_coarse_to_fine(E, N, U, start_point, rays_2d,
                               delta_steps=(5.0, 1.0), max_iters=(4096, 4096),
                               dtype=dtype_r):
    """
    rays_2d: (3, Nr)
    Returns r: (Nr,)
    """
    coarse_delta = jnp.asarray(delta_steps[0], dtype=dtype)
    fine_delta   = jnp.asarray(delta_steps[1], dtype=dtype)
    coarse_iter  = int(max_iters[0])
    fine_iter    = int(max_iters[1])

    r_coarse = ray_trace_basic_jax_jit(
        E, N, U, start_point, rays_2d,
        delta_r_m=coarse_delta, max_iter=coarse_iter
    )
    r_start = jnp.where(
        jnp.isnan(r_coarse),
        jnp.nan,
        jnp.maximum(r_coarse - coarse_delta, fine_delta),
    )
    r_fine = ray_trace_basic_jax_jit(
        E, N, U, start_point, rays_2d,
        delta_r_m=fine_delta, r_start=r_start, max_iter=fine_iter
    )
    return jnp.where(jnp.isnan(r_coarse), jnp.nan, r_fine)

ray_distance_coarse_to_fine_jit = jax.jit(
    ray_distance_coarse_to_fine,
    static_argnames=("delta_steps", "max_iters"),
)
