import jax
import jax.numpy as jnp
import jax_cosmo as jc
from jax_cosmo import Cosmology

from jaxpm.kernels import fftk, gradient_kernel, invlaplace_kernel, invnabla_kernel, longrange_kernel
from jaxpm.painting import cic_paint, cic_read

from jaxpm.graph import get_graph, get_graph_given_edges


def hpm_forces(
    scale,
    dm_pos,
    gas_pos,
    mesh_shape,
    cosmo,
    model,
    gas_latent=None,
    gravity_only=False,
    r_split=0,
    architecture="mlp",
    edges=None,
):
    kvec = fftk(mesh_shape)

    N_dm = cic_paint(jnp.zeros(mesh_shape), dm_pos)
    N_gas = cic_paint(jnp.zeros(mesh_shape), gas_pos)
    gas_N = cic_read(N_gas, gas_pos)

    # assume identical mass for all particles [dm_mass particle]
    rho_dm = N_dm
    rho_gas = N_gas * (cosmo.Omega_b / cosmo.Omega_c)
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
    if not gravity_only:
        gas_rho = cic_read(rho_gas, gas_pos)

        rho_gas_k = jnp.fft.rfftn(rho_gas)
        fscalar_gas = jnp.fft.irfftn(rho_gas_k * invnabla_kernel(kvec))
        gas_fscalar = cic_read(fscalar_gas, gas_pos)

        # gas_rho = cic_read(rho_tot, gas_pos)
        # gas_fscalar = cic_read(jnp.fft.irfftn(rho_k_tot * invnabla_kernel(kvec)), gas_pos)

        if architecture == "mlp":
            gas_inputs = jnp.stack(
                [
                    jnp.tile(scale, gas_pos.shape[0]),
                    jnp.log10(gas_rho),
                    jnp.arcsinh(gas_fscalar / 100),
                ],
                axis=-1,
            )
            if gas_latent is None:
                print("No latent variable")
                gas_P = 10 ** jnp.squeeze(model(gas_inputs))
            else:
                print("With latent variable")
                gas_inputs = jnp.concatenate([gas_inputs, jnp.expand_dims(gas_latent, axis=-1)], axis=-1)
                gas_preds = model(gas_inputs)
                gas_P, gas_latent = 10 ** gas_preds[:, 0], gas_preds[:, 1]

        elif architecture == "mlp+cnn":
            particle_input = jnp.stack(
                [jnp.tile(scale, gas_pos.shape[0]), jnp.log10(gas_rho), jnp.arcsinh(gas_fscalar / 100)], axis=-1
            )
            field_input = jnp.stack([jnp.log10(rho_tot + 1), jnp.arcsinh(fscalar_tot / 100)], axis=-1)
            gas_P = 10 ** jnp.squeeze(model(gas_pos, particle_input, field_input))

        elif architecture == "gnn":
            if edges is None:
                print("On-the-fly graph")
                graph = jax.lax.stop_gradient(get_graph(scale, gas_pos, gas_rho, gas_fscalar))
            else:
                print("Prebuilt graph")
                graph = get_graph_given_edges(scale, edges, gas_rho, gas_fscalar)
            gas_P = 10 ** jnp.squeeze(model(graph).nodes)

        else:
            raise ValueError(f"Unknown model type {architecture}")

        P_gas = cic_paint(jnp.zeros(mesh_shape), gas_pos, weight=gas_P / gas_N)
        P_gas_k = jnp.fft.rfftn(P_gas)

        def pressure(pos):
            nabla_P = jnp.stack(
                [cic_read(jnp.fft.irfftn(gradient_kernel(kvec, i) * P_gas_k), pos) for i in range(len(kvec))],
                axis=-1,
            )
            return nabla_P / jnp.expand_dims(gas_rho, axis=-1)

        gas_force -= pressure(gas_pos)

    return dm_force, gas_force


def hpm_table_forces_temp(
    scale, dm_pos, gas_pos, mesh_shape, cosmo, model, params=None, gravity_only=False, r_split=0
):
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
        gas_P /= 1000

        gas_rho = cic_read(rho_gas, gas_pos)
        P_gas = cic_paint(jnp.zeros(mesh_shape), gas_pos, weight=gas_P / gas_rho)
        P_k_gas = jnp.fft.rfftn(P_gas)

        def pressure(pos):
            nabla_P = jnp.stack(
                [cic_read(jnp.fft.irfftn(gradient_kernel(kvec, i) * P_k_gas), pos) for i in range(len(kvec))],
                axis=-1,
            )
            # return nabla_P / jnp.expand_dims(gas_rho, axis=-1)
            return nabla_P

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

    dm_force = -gravity(dm_pos)
    gas_force = -gravity(gas_pos)

    if not gravity_only:
        gas_rho = cic_read(rho_gas, gas_pos)

        rho_gas_k = jnp.fft.rfftn(rho_gas)
        fscalar_gas = jnp.fft.irfftn(rho_gas_k * invnabla_kernel(kvec))
        gas_fscalar = cic_read(fscalar_gas, gas_pos)

        gas_inputs = jnp.stack(
            [
                jnp.tile(scale, gas_pos.shape[0]),
                jnp.log10(gas_rho),
                jnp.arcsinh(gas_fscalar / 100),
            ],
            axis=-1,
        )

        # def pressure(pos):
        #     return jnp.stack(
        #         [
        #             cic_read(jnp.fft.irfftn(gradient_kernel(kvec, i) * model(k, jnp.atleast_1d(scale))), pos)
        #             for i in range(len(kvec))
        #         ],
        #         axis=-1,
        #     )

        # gas_force += 10 ** model(gas_inputs)
        gas_force += model(gas_inputs)
    # # pressure force
    # k = jnp.sqrt(sum((ki / jnp.pi) ** 2 for ki in kvec))

    # fscalar_k = rho_k_tot * invnabla_kernel(kvec)

    # def pressure(pos):
    #     return jnp.stack(
    #         [
    #             cic_read(jnp.fft.irfftn(gradient_kernel(kvec, i) * fscalar_k * model(k, jnp.atleast_1d(scale))), pos)
    #             for i in range(len(kvec))
    #         ],
    #         axis=-1,
    #     )

    # dm_force = -gravity(dm_pos)
    # gas_force = -gravity(gas_pos)
    # if not gravity_only:
    #     gas_force -= pressure(gas_pos)

    return dm_force, gas_force


def get_hpm_network_ode_fn(
    model,
    mesh_shape,
    cosmo: Cosmology,
    integrator_type: str = "odeint",
    architecture: str = "mlp",
    precomputed_edges=None,
    gravity_only=False,
):
    def hpm_ode(scale, state, kwargs):
        dm_pos, dm_vel, gas_pos, gas_vel = state

        # dm_force, gas_force = hpm_direct_forces(
        #     scale,
        #     dm_pos,
        #     gas_pos,
        #     mesh_shape,
        #     cosmo,
        #     model,
        #     gravity_only=gravity_only,
        # )

        dm_force, gas_force = hpm_forces(
            scale,
            dm_pos,
            gas_pos,
            mesh_shape,
            cosmo,
            model,
            gravity_only=gravity_only,
            architecture=architecture,
            edges=precomputed_edges,
            **kwargs,
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
        ode_fn = lambda state, scale, args: hpm_ode(scale, state, args)
    elif integrator_type == "diffrax":
        ode_fn = lambda scale, state, args: hpm_ode(scale, state, args)
    else:
        raise ValueError(f"Unknown integrator type {integrator_type}")

    return ode_fn
