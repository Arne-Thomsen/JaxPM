import jax
import jax.numpy as jnp
import jax_cosmo as jc
from jax_cosmo import Cosmology

from jaxpm.kernels import fftk, gradient_kernel, invlaplace_kernel, invnabla_kernel, longrange_kernel
from jaxpm.painting import cic_paint, cic_read


def hpm_table_forces(scale, dm_pos, gas_pos, mesh_shape, cosmo, model, params=None, gravity_only=False, r_split=0):
    kvec = fftk(mesh_shape)

    rho_dm = cic_paint(jnp.zeros(mesh_shape), dm_pos)
    rho_gas = cic_paint(jnp.zeros(mesh_shape), gas_pos, weight=cosmo.Omega_b / cosmo.Omega_c)
    rho_tot = rho_dm + rho_gas

    # gravitational potential
    rho_k_tot = jnp.fft.rfftn(rho_tot)
    phi_k_tot = rho_k_tot * invlaplace_kernel(kvec) * longrange_kernel(kvec, r_split=r_split)

    def gravity(pos):
        return jnp.stack(
            [cic_read(jnp.fft.irfftn(gradient_kernel(kvec, i) * phi_k_tot), pos) for i in range(len(kvec))],
            axis=-1,
        )

    dm_force = -gravity(dm_pos)
    gas_force = -gravity(gas_pos)

    # pressure force
    if not gravity_only:
        gas_rho_tot = cic_read(rho_tot, gas_pos)
        gas_fscalar = cic_read(jnp.fft.irfftn(rho_k_tot * invnabla_kernel(kvec)), gas_pos)
        gas_inputs = jnp.stack(
            [jnp.tile(scale, gas_pos.shape[0]), jnp.log10(gas_rho_tot), jnp.arcsinh(gas_fscalar / 100)], axis=-1
        )
        gas_preds = model(gas_inputs)
        gas_P, gas_T = gas_preds[:, 0], gas_preds[:, 1]

        # if params is not None:
        #     gas_P += params["a"] + jnp.log10(scale) * params["b"]
        #     # gas_P += params["m1"] * jnp.log10(scale) + params["b"]
        #     # gas_P += params["m1"] * jnp.sqrt(scale) + params["b"]
        #     # gas_P += params["m1"] * scale + params["m2"] * scale**2 + params["b"]

        gas_P, gas_T = 10**gas_P, 10**gas_T

        # gas_P /= 64880627
        # gas_P /= (cosmo.Omega_b / cosmo.Omega_c) * 64880627
        # gas_P /= scale ** (1 / 2)
        # gas_P /= jnp.sqrt(scale)
        # gas_P *= scale**5
        # gas_P /= scale
        # gas_P /= (2 * jnp.pi) ** 3
        # gas_P /= 1000

        gas_rho = cic_read(rho_gas, gas_pos)
        P_gas = cic_paint(jnp.zeros(mesh_shape), gas_pos, weight=gas_P / gas_rho)
        P_k_gas = jnp.fft.rfftn(P_gas)

        def pressure(pos):
            nabla_P = jnp.stack(
                [cic_read(jnp.fft.irfftn(gradient_kernel(kvec, i) * P_k_gas), pos) for i in range(len(kvec))],
                axis=-1,
            )
            return nabla_P / jnp.expand_dims(gas_rho, axis=-1)

        gas_force -= pressure(gas_pos)

    return dm_force, gas_force


def hpm_table_forces_temp(scale, dm_pos, gas_pos, mesh_shape, cosmo, model, gravity_only=False, r_split=0):
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

    dm_force = -gravity(dm_pos)
    gas_force = -gravity(gas_pos)

    # pressure force
    # jax.debug.print("Scale: {}", scale)
    if not gravity_only:
        gas_rho_tot = cic_read(rho_tot, gas_pos)
        gas_fscalar = cic_read(jnp.fft.irfftn(rho_k_tot * invnabla_kernel(kvec)), gas_pos)
        gas_inputs = jnp.stack(
            [jnp.tile(scale, gas_pos.shape[0]), jnp.log10(gas_rho_tot), jnp.arcsinh(gas_fscalar / 100)], axis=-1
        )
        # gas_inputs = jnp.stack(
        #     [jnp.log10(gas_rho_tot), jnp.arcsinh(gas_fscalar / 10)],
        #     axis=-1,
        # )
        gas_P = 10 ** jnp.squeeze(model(gas_inputs))
        # gas_P = jnp.squeeze(model(gas_inputs))

        gas_rho = cic_read(rho_gas, gas_pos)
        P_k = jnp.fft.rfftn(cic_paint(jnp.zeros(mesh_shape), gas_pos, weight=gas_P / gas_rho))
        # print(P_k.shape)

        def pressure(pos):
            nabla_P = jnp.stack(
                [cic_read(jnp.fft.irfftn(gradient_kernel(kvec, i) * P_k), pos) for i in range(len(kvec))],
                axis=-1,
            )
            return nabla_P / jnp.expand_dims(gas_rho, axis=-1)

        gas_force -= pressure(gas_pos)

    return dm_force, gas_force


def hpm_direct_forces(scale, dm_pos, gas_pos, mesh_shape, cosmo, model, gravity_only=False, r_split=0):
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
    k = jnp.sqrt(sum((ki / jnp.pi) ** 2 for ki in kvec))

    fscalar_k = rho_k_tot * invnabla_kernel(kvec)

    def pressure(pos):
        return jnp.stack(
            [
                cic_read(jnp.fft.irfftn(gradient_kernel(kvec, i) * fscalar_k * model(k, jnp.atleast_1d(scale))), pos)
                for i in range(len(kvec))
            ],
            axis=-1,
        )

    dm_force = -gravity(dm_pos)
    gas_force = -gravity(gas_pos)
    if not gravity_only:
        gas_force -= pressure(gas_pos)

    return dm_force, gas_force


def get_hpm_network_ode_fn(
    model,
    mesh_shape,
    cosmo: Cosmology,
    integrator_type: str = "odeint",
    force_type: str = "table",
    gravity_only=False,
):
    def hpm_ode(scale, state):
        dm_pos, dm_vel, gas_pos, gas_vel = state

        if force_type == "table":
            dm_force, gas_force = hpm_table_forces(
                scale, dm_pos, gas_pos, mesh_shape, cosmo, model, gravity_only=gravity_only
            )
        elif force_type == "direct":
            dm_force, gas_force = hpm_direct_forces(
                scale, dm_pos, gas_pos, mesh_shape, cosmo, model, gravity_only=gravity_only
            )

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
        ode_fn = lambda scale, state, args: hpm_ode(scale, state)
    else:
        raise ValueError(f"Unknown integrator type {integrator_type}")

    return ode_fn
