import jax.numpy as jnp
import jax_cosmo as jc
from jax_cosmo import Cosmology

from jaxpm.kernels import fftk, gradient_kernel, invlaplace_kernel, invnabla_kernel, longrange_kernel
from jaxpm.painting import cic_paint, cic_read


def hpm_forces(scale, dm_pos, gas_pos, mesh_shape, cosmo, model, r_split=0):
    kvec = fftk(mesh_shape)

    rho_dm = cic_paint(jnp.zeros(mesh_shape), dm_pos)
    rho_gas = cic_paint(jnp.zeros(mesh_shape), gas_pos, weight=cosmo.Omega_b / cosmo.Omega_c)
    rho_tot = rho_dm + rho_gas

    # gravitational potential
    rho_k_tot = jnp.fft.rfftn(rho_tot)
    phi_k = rho_k_tot * invlaplace_kernel(kvec) * longrange_kernel(kvec, r_split=r_split)

    def gravity(pos):
        return jnp.stack(
            [cic_read(jnp.fft.irfftn(gradient_kernel(kvec, i) * phi_k), pos) for i in range(len(kvec))],
            axis=-1,
        )

    # pressure force
    gas_rho_tot = cic_read(rho_tot, gas_pos)
    gas_fscalar = cic_read(jnp.fft.irfftn(rho_k_tot * invnabla_kernel(kvec)), gas_pos)
    gas_inputs = jnp.stack(
        [jnp.tile(scale, gas_pos.shape[0]), jnp.log10(gas_rho_tot), jnp.arcsinh(gas_fscalar / 100)], axis=-1
    )
    gas_preds = model(gas_inputs)
    gas_P, gas_T = 10 ** gas_preds[:, 0], 10 ** gas_preds[:, 1]

    # TODO
    # gas_P = gas_P / (1e10 * (1e3) ** 3)
    gas_P /= 1e10
    # gas_P /= 1e19

    gas_rho = cic_read(rho_gas, gas_pos)
    P_k = jnp.fft.rfftn(cic_paint(jnp.zeros(mesh_shape), gas_pos, weight=gas_P / gas_rho))

    def pressure(pos):
        nabla_P = jnp.stack(
            [cic_read(jnp.fft.irfftn(gradient_kernel(kvec, i) * P_k), pos) for i in range(len(kvec))],
            axis=-1,
        )
        return nabla_P / jnp.expand_dims(gas_rho, axis=-1)

    dm_force = -gravity(dm_pos)
    gas_force = -gravity(gas_pos) - pressure(gas_pos)

    return dm_force, gas_force


def get_hpm_network_ode_fn(model, mesh_shape, cosmo: Cosmology, integrator_type: str = "odeint"):
    def hpm_ode(scale, state):
        dm_pos, dm_vel, gas_pos, gas_vel = state

        dm_force, gas_force = hpm_forces(scale, dm_pos, gas_pos, mesh_shape, cosmo, model)
        # TODO double check these factors
        dm_force *= 1.5 * cosmo.Omega_m
        gas_force *= 1.5 * cosmo.Omega_m

        # update the positions (drift)
        pos_fac = 1.0 / (scale**3 * jnp.sqrt(jc.background.Esqr(cosmo, scale)))
        d_dm_pos = pos_fac * dm_vel
        d_gas_pos = pos_fac * gas_vel

        # update the velocities (kick)
        vel_fac = 1.0 / (scale**2 * jnp.sqrt(jc.background.Esqr(cosmo, scale)))
        d_dm_vel = vel_fac * dm_force
        d_gas_vel = vel_fac * gas_force

        return jnp.stack([d_dm_pos, d_dm_vel, d_gas_pos, d_gas_vel])

    if integrator_type == "odeint":
        ode_fn = lambda state, scale: hpm_ode(scale, state)
    elif integrator_type == "diffrax":
        ode_fn = hpm_ode
    else:
        raise ValueError(f"Unknown integrator type {integrator_type}")

    return ode_fn
