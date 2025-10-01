# taken from https://github.com/DifferentiableUniverseInitiative/Baryonator/blob/main/notebooks/EGD_demo_on_CAMELS.ipynb
# by Francois
import jax
import jax.numpy as jnp
import optax

from jaxpm.painting import cic_read, cic_paint
from jaxpm.kernels import fftk, gaussian_kernel, gradient_kernel


def apply_egd_correction(params, dmo_dm_pos, cosmo, mesh_shape):
    dmo_rho_tot = cic_paint(jnp.zeros(mesh_shape), dmo_dm_pos)
    dmo_delta_tot = dmo_rho_tot / dmo_rho_tot.mean() - 1

    inds = jax.random.permutation(jax.random.PRNGKey(11), jnp.arange(0, len(dmo_dm_pos) - 1))
    split = int(cosmo.Omega_b / cosmo.Omega_m * len(dmo_dm_pos))
    egd_gas_pos = dmo_dm_pos[inds[:split]]
    egd_dm_pos = dmo_dm_pos[inds[split:]]

    egd_gas_pos += egd_correction(params, dmo_delta_tot, egd_gas_pos, mesh_shape)

    egd_rho_dm = cic_paint(jnp.zeros(mesh_shape), egd_dm_pos)
    egd_rho_gas = cic_paint(jnp.zeros(mesh_shape), egd_gas_pos)

    # the Om / Ob weighting is implicit in the particle counts
    egd_rho_tot = egd_rho_dm + egd_rho_gas

    return egd_gas_pos, egd_dm_pos, egd_rho_dm, egd_rho_gas, egd_rho_tot


def egd_correction(params, delta, pos, mesh_shape):
    """
    Will compute the EGD displacement as a function of density traced by the
    input particles.
    params contains [amplitude, scale, gamma]
    """

    kvec = fftk(mesh_shape)
    alpha, kl, gamma = params

    # Compute a temperature-like map from density contrast
    T = (delta + 1) ** gamma

    # Apply FFT to apply filtering
    T_k = jnp.fft.rfftn(T)
    filtered_T_k = gaussian_kernel(kvec, kl) * T_k  # This applies a Gaussian smoothing

    # Compute derivatives of this filtered T-like field at the position of particles
    dpos_egd = jnp.stack(
        [cic_read(jnp.fft.irfftn(gradient_kernel(kvec, i) * filtered_T_k), pos) for i in range(3)], axis=-1
    )

    # Apply overal scaling of these displacements
    dpos_egd = -alpha * dpos_egd

    return dpos_egd


# def egd_loss_fn(params):
#     egdized_pos_gas = egd_pos_gas + egd_correction(total_delta_dmo, egd_pos_gas, params)

#     # Add the gas that we have moved to the dm that stayed where it was
#     delta = cic_paint(egd_rho_dm, egdized_pos_gas)
#     delta = delta / delta.mean() - 1.0

#     # Compute the PS of everybody
#     k, pk = p_spectrum(delta)

#     # Some weighting function of the pk, following Biwei's strategy
#     inds = k < 10.0
#     return jnp.sum(((pk / pk_hydro - 1))[inds] ** 2, axis=-1)
# return jnp.sum( (delta-total_delta_hydro)**2)


def update(params, optimizer, opt_state, loss_fn):
    """Single SGD update step."""

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return loss, new_params, new_opt_state
