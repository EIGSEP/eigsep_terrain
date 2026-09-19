'''Jittable joint logL over all images, the JAX analog of
PositionSolver.total_logL (horizon-ray and antenna terms only; tie-point
and GPS-delta priors are not ported).

Usage::

    problem = stack_problem(solver.export_jax())
    logL = logL_from_problem_jit(theta, problem)

where theta is the same flat vector PositionSolver.set_mcmc_prms takes:
7 PRM_ORDER values per fit image, followed by the antenna (e, n, u).
'''
import jax
import jax.numpy as jnp
from .img_jax import horizon_ray_logL_jax, ant_logL_jax

N_PRM = 7  # len(img.PRM_ORDER); not imported to avoid img.py's deps
dtype_r = jnp.float32


def unpack_theta(theta, n_fit, dtype=dtype_r):
    theta = jnp.asarray(theta, dtype=dtype)
    img_prms = theta[:n_fit * N_PRM].reshape(n_fit, N_PRM)
    ant = theta[n_fit * N_PRM:n_fit * N_PRM + 3]
    return img_prms, ant


def logL_from_problem(theta, problem, eps=1e-3, max_iters=(4096, 4096)):
    dem = problem['dem']
    E, N, U = dem['E'], dem['N'], dem['U']
    fit = problem['fit']
    all_imgs = problem['all']
    box_size = problem['box_size']
    n_fit = fit['Nu'].shape[0]

    img_prms, (ant_e, ant_n, ant_u) = unpack_theta(theta, n_fit)

    # horizon term over fit images
    def fit_one(prm, Nu, Nv, x_px, y_px, psky, corr_px):
        e, n, u, th, ph, ti, f = prm
        return horizon_ray_logL_jax(E, N, U, Nu, Nv,
                                    e, n, u, th, ph, ti, f,
                                    x_px, y_px, psky, eps=eps,
                                    correlation_px=corr_px,
                                    max_iters=max_iters)

    logL_rays = jnp.sum(jax.vmap(fit_one)(
        img_prms, fit['Nu'], fit['Nv'], fit['x_px'], fit['y_px'],
        fit['psky'], fit['correlation_px']))

    # antenna term over all images: fit images take their pose from theta,
    # static images keep their exported pose
    def pick_prm(is_fit, fit_index, fixed_prm):
        fit_index_safe = jnp.clip(fit_index, 0, n_fit - 1)
        return jnp.where(is_fit, img_prms[fit_index_safe], fixed_prm)

    prms_all = jax.vmap(pick_prm)(all_imgs['is_fit'],
                                  all_imgs['fit_index'],
                                  all_imgs['fixed_prms'])

    def ant_one(prm, Nu, Nv, u0, v0):
        e, n, u, th, ph, ti, f = prm
        return ant_logL_jax(Nu, Nv, e, n, u, th, ph, ti, f,
                            ant_e, ant_n, ant_u, u0, v0, box_size)

    logL_ant = jax.vmap(ant_one)(prms_all, all_imgs['Nu'], all_imgs['Nv'],
                                 all_imgs['ant_u_px'], all_imgs['ant_v_px'])
    logL_ant = jnp.sum(jnp.where(all_imgs['has_ant'], logL_ant, 0.0))

    return logL_rays + logL_ant


logL_from_problem_jit = jax.jit(logL_from_problem,
                                static_argnames=('max_iters',))


def stack_problem(export):
    '''Stack the per-image dicts from PositionSolver.export_jax() into the
    batched arrays logL_from_problem expects. Fit images must share n_rays
    so their pixel arrays stack.'''
    fit_exports, all_exports = export['fit'], export['all']
    n_fit = len(fit_exports)
    # PositionSolver.imgs = fit_imgs + static_imgs
    fit_index = [i if i < n_fit else -1 for i in range(len(all_exports))]

    def stack(dicts, key, dtype=None):
        return jnp.asarray([d[key] for d in dicts], dtype=dtype)

    fit = {
        'Nu': stack(fit_exports, 'npix_y', jnp.int32),
        'Nv': stack(fit_exports, 'npix_x', jnp.int32),
        'x_px': stack(fit_exports, 'x_px', jnp.int32),
        'y_px': stack(fit_exports, 'y_px', jnp.int32),
        'psky': stack(fit_exports, 'psky', dtype_r),
        'correlation_px': stack(fit_exports, 'px_smooth', dtype_r),
    }
    ant_px = stack(all_exports, 'ant_px', dtype_r)
    all_ = {
        'Nu': stack(all_exports, 'npix_y', jnp.int32),
        'Nv': stack(all_exports, 'npix_x', jnp.int32),
        'ant_u_px': ant_px[:, 0],
        'ant_v_px': ant_px[:, 1],
        'has_ant': stack(all_exports, 'has_ant', jnp.bool_),
        'fixed_prms': stack(all_exports, 'prms', dtype_r),
        'is_fit': jnp.asarray([i >= 0 for i in fit_index], dtype=jnp.bool_),
        'fit_index': jnp.asarray(fit_index, dtype=jnp.int32),
    }
    return {
        'dem': {k: jnp.asarray(export['dem'][k]) for k in ('E', 'N', 'U')},
        'fit': fit,
        'all': all_,
        'box_size': jnp.asarray(export['box_size'], dtype=dtype_r),
    }
