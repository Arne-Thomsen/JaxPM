import jax
import jax.numpy as jnp
import jax_cosmo as jc
from jax_cosmo import Cosmology

from jaxpm.kernels import fftk, gradient_kernel, invlaplace_kernel, invnabla_kernel, longrange_kernel
from jaxpm.painting import cic_paint, cic_read
from jaxpm.graph import get_graph_given_edges, get_graph_from_features
from jaxpm.data import get_hpm_inputs


def hpm_forces(
    mesh_per_dim,
    cosmo,
    scale,
    dm_pos,
    gas_pos=None,
    # gravity
    gravity_model=None,
    r_split=0,
    # pressure
    pressure_model=None,
    gas_vel=None,
    gas_latent=None,
    gas_architecture="mlp",
    graph_edges=None,
    graph_kwargs={},
    training=False,
):
    mesh_shape = [mesh_per_dim] * 3

    N_dm = cic_paint(jnp.zeros(mesh_shape), dm_pos)
    if with_gas := gas_pos is not None:
        N_gas = cic_paint(jnp.zeros(mesh_shape), gas_pos)
        # TODO could also initialize with weights from the simulation
        # assume identical mass for all particles of a given species
        rho_dm = N_dm * cosmo.Omega_c / (cosmo.Omega_c + cosmo.Omega_b)
        rho_gas = N_gas * cosmo.Omega_b / (cosmo.Omega_c + cosmo.Omega_b)
        rho_tot = rho_dm + rho_gas
    else:
        rho_tot = N_dm

    # necessary for mesh_per_dim != parts_per_dim
    delta_tot = rho_tot / jnp.mean(rho_tot) - 1

    # gravitational potential
    kvec = fftk(mesh_shape)
    delta_k_tot = jnp.fft.rfftn(delta_tot)
    phi_k_tot = delta_k_tot * invlaplace_kernel(kvec) * longrange_kernel(kvec, r_split=r_split)
    phi_k_tot *= 1.5 * cosmo.Omega_m

    if gravity_model is not None:
        print(f"Using learned correction to the gravitational potential")
        k = jnp.sqrt(sum((ki / jnp.pi) ** 2 for ki in kvec))
        phi_k_tot += phi_k_tot * gravity_model(k, jnp.atleast_1d(scale))

    def gravity(pos):
        return jnp.stack(
            [cic_read(jnp.fft.irfftn(gradient_kernel(kvec, i) * phi_k_tot), pos) for i in range(len(kvec))],
            axis=-1,
        )

    if with_gas:
        # TODO
        # dm_force = -gravity(dm_pos) * cosmo.Omega_c / (cosmo.Omega_c + cosmo.Omega_b)
        # gas_force = -gravity(gas_pos) * cosmo.Omega_b / (cosmo.Omega_c + cosmo.Omega_b)
        dm_force = -gravity(dm_pos)
        gas_force = -gravity(gas_pos)
    else:
        dm_force = -gravity(dm_pos)
        gas_force = None

    # pressure force
    d_gas_latent = 0.0
    if with_pressure := pressure_model is not None:
        print(f"Using learned pressure force")

        gas_N = cic_read(N_gas, gas_pos)
        gas_rho = cic_read(rho_gas, gas_pos)

        # particle-level network output
        if gas_architecture in ["mlp", "mlp+cnn", "gnn", "offline"]:
            if gas_architecture == "mlp":
                gas_inputs = get_hpm_inputs(
                    scale,
                    gas_pos,
                    gas_vel,
                    gas_rho,
                    rho_gas,
                    gas_N,
                    mesh_shape,
                    gas_latent=gas_latent,
                    return_vel=True,
                    return_field=False,
                )
                gas_preds = pressure_model(gas_inputs, training=training)
            elif gas_architecture == "mlp+cnn":
                gas_inputs, field_inputs = get_hpm_inputs(
                    scale,
                    gas_pos,
                    gas_vel,
                    gas_rho,
                    rho_gas,
                    gas_N,
                    mesh_shape,
                    gas_latent=gas_latent,
                    return_field=True,
                )
                gas_preds = pressure_model(gas_pos, gas_inputs, field_inputs, training=training)
            elif gas_architecture == "gnn":
                gas_inputs = get_hpm_inputs(
                    scale,
                    gas_pos,
                    gas_vel,
                    gas_rho,
                    rho_gas,
                    gas_N,
                    mesh_shape,
                    gas_latent=gas_latent,
                    return_field=False,
                )
                if graph_edges is None:
                    print("On-the-fly graph")
                    graph = get_graph_from_features(gas_inputs, scale, **graph_kwargs)
                else:
                    print("Prebuilt graph")
                    graph = get_graph_given_edges(gas_inputs, graph_edges, current_scale=scale)
                gas_preds = pressure_model(graph, training=training)
            elif gas_architecture == "offline":
                gas_inputs = get_hpm_inputs(
                    scale,
                    gas_pos,
                    gas_vel,
                    gas_rho,
                    rho_gas,
                    gas_N,
                    mesh_shape,
                    gas_latent=gas_latent,
                    return_vel=True,
                    return_field=False,
                )
                # gas_preds = gas_model(gas_inputs) - 2
                gas_preds = pressure_model(gas_inputs) - 1.7
                # gas_preds = gas_model(gas_inputs) - 3
                # gas_preds = gas_model(gas_inputs)
            else:
                raise ValueError(f"Unknown model type {gas_architecture}")

            if gas_latent is None:
                print("No latent variable")
                gas_P = 10 ** jnp.squeeze(gas_preds)

                # gas_U = 10 ** jnp.squeeze(gas_preds)
                # gas_P = 2 / 3 * gas_U * gas_rho
            else:
                print(f"With latent variable")
                gas_P, d_gas_latent = 10 ** gas_preds[:, 0], gas_preds[:, 1:]

                # gas_U, d_gas_latent = 10 ** gas_preds[:, 0], gas_preds[:, 1:]
                # gas_P = 2 / 3 * gas_U * gas_rho

            P_gas = cic_paint(jnp.zeros(mesh_shape), gas_pos, weight=gas_P / gas_N)

        # field-level network output
        else:
            if gas_architecture == "cnn":
                _, field_inputs = get_hpm_inputs(
                    scale,
                    gas_pos,
                    gas_vel,
                    gas_rho,
                    rho_gas,
                    gas_N,
                    mesh_shape,
                    latent_gas=gas_latent,
                    return_field=True,
                )
                preds_gas = pressure_model(field_inputs, jnp.atleast_1d(scale), training=training)
            else:
                raise ValueError(f"Unknown model type {gas_architecture}")

            if gas_latent is None:
                print("No latent variable")
                P_gas = 10 ** jnp.squeeze(preds_gas)
            else:
                print(f"With latent variable")
                P_gas, d_gas_latent = 10 ** preds_gas[..., 0], preds_gas[..., 1:]

        # d_gas_latent -= jnp.mean(d_gas_latent)
        P_gas_k = jnp.fft.rfftn(P_gas)

        def pressure(pos):
            nabla_P = jnp.stack(
                [cic_read(jnp.fft.irfftn(gradient_kernel(kvec, i) * P_gas_k), pos) for i in range(len(kvec))],
                axis=-1,
            )
            return nabla_P / jnp.expand_dims(gas_rho, axis=-1)

        gas_force -= pressure(gas_pos)

    return dm_force, gas_force, d_gas_latent


def hpm_forces_denise(
    mesh_per_dim,
    cosmo,
    scale,
    dm_pos,
    gas_pos=None,
    gas_mass_residual=None,
    # gravity
    gravity_model=None,
    r_split=0,
    # pressure
    pressure_model=None,
    fourier_model=None,
    gas_vel=None,
    gas_latent=None,
    gas_architecture="mlp",
    graph_edges=None,
    graph_kwargs={},
):
    print("Using Denise style correction")
    mesh_shape = [mesh_per_dim] * 3

    N_dm = cic_paint(jnp.zeros(mesh_shape), dm_pos)
    if with_gas := gas_pos is not None:
        N_gas = cic_paint(jnp.zeros(mesh_shape), gas_pos)
        # TODO could also initialize with weights from the simulation
        # assume identical mass for all particles of a given species
        rho_dm = N_dm * cosmo.Omega_c / (cosmo.Omega_c + cosmo.Omega_b)

        rho_gas = N_gas * cosmo.Omega_b / (cosmo.Omega_c + cosmo.Omega_b)

        # if gas_mass_residual is not None:
        #     rho_gas = cic_paint(
        #         jnp.zeros(mesh_shape),
        #         gas_pos,
        #         gas_mass_residual / 1e3 + cosmo.Omega_b / (cosmo.Omega_c + cosmo.Omega_b),
        #         # cosmo.Omega_b / (cosmo.Omega_c + cosmo.Omega_b),
        #     )
        #     print("using variable gas mass")
        # else:
        #     rho_gas = N_gas * cosmo.Omega_b / (cosmo.Omega_c + cosmo.Omega_b)

        rho_tot = rho_dm + rho_gas
    else:
        rho_tot = N_dm

    # necessary for mesh_per_dim != parts_per_dim
    delta_tot = rho_tot / jnp.mean(rho_tot) - 1

    # gravitational potential
    kvec = fftk(mesh_shape)
    delta_k_tot = jnp.fft.rfftn(delta_tot)
    phi_k_tot = delta_k_tot * invlaplace_kernel(kvec) * longrange_kernel(kvec, r_split=r_split)
    phi_k_tot *= 1.5 * cosmo.Omega_m

    if gravity_model is not None:
        print(f"Using learned correction to the gravitational potential")
        k = jnp.sqrt(sum((ki / jnp.pi) ** 2 for ki in kvec))
        phi_k_tot += phi_k_tot * gravity_model(k, jnp.atleast_1d(scale))

    def gravity(pos):
        return jnp.stack(
            [cic_read(jnp.fft.irfftn(gradient_kernel(kvec, i) * phi_k_tot), pos) for i in range(len(kvec))],
            axis=-1,
        )

    if with_gas:
        dm_force = -gravity(dm_pos)
        gas_force = -gravity(gas_pos)
    else:
        dm_force = -gravity(dm_pos)
        gas_force = None

    d_gas_latent = 0.0
    if pressure_model is not None:
        print(f"Using Denise-style learned pressure force")
        gas_N = cic_read(N_gas, gas_pos)
        gas_rho = cic_read(rho_gas, gas_pos)

        gas_inputs = get_hpm_inputs(
            scale,
            gas_pos,
            gas_vel,
            gas_rho,
            rho_gas,
            gas_N,
            mesh_shape,
            gas_latent=gas_latent,
            return_vel=True,
            return_field=False,
        )
        # gas_preds = pressure_model(gas_inputs)
        gas_preds = pressure_model(gas_inputs) - 2
        gas_P = 10 ** jnp.squeeze(gas_preds)

        P_gas = cic_paint(jnp.zeros(mesh_shape), gas_pos, weight=gas_P / gas_N)

        k = jnp.sqrt(sum((ki / jnp.pi) ** 2 for ki in kvec))

        P_gas_k = jnp.fft.rfftn(P_gas)
        P_gas_k += P_gas_k * fourier_model(k, jnp.atleast_1d(scale))

        # P_gas_k = jnp.fft.rfftn(P_gas) * fourier_model(k, jnp.atleast_1d(scale))

        def pressure(pos):
            nabla_P = jnp.stack(
                [cic_read(jnp.fft.irfftn(gradient_kernel(kvec, i) * P_gas_k), pos) for i in range(len(kvec))],
                axis=-1,
            )
            return nabla_P / jnp.expand_dims(gas_rho, axis=-1)

        gas_force -= pressure(gas_pos)

    # d_gas_latent = 0.0
    # if pressure_model is not None:
    #     print(f"Using Denise-style learned pressure force")

    #     delta_gas = rho_gas / jnp.mean(rho_gas) - 1
    #     delta_k_gas = jnp.fft.rfftn(delta_gas)
    #     phi_k_gas = delta_k_gas * invlaplace_kernel(kvec) * longrange_kernel(kvec, r_split=r_split)

    #     k = jnp.sqrt(sum((ki / jnp.pi) ** 2 for ki in kvec))
    #     phi_k_gas += phi_k_gas * pressure_model(k, jnp.atleast_1d(scale))

    #     def pressure(pos):
    #         return jnp.stack(
    #             [cic_read(jnp.fft.irfftn(gradient_kernel(kvec, i) * phi_k_gas), pos) for i in range(len(kvec))],
    #             axis=-1,
    #         )
    #         return

    #     gas_force -= pressure(gas_pos)

    return dm_force, gas_force, d_gas_latent
    # return dm_force, gas_force
    # return dm_force, gas_force, d_gas_mass


def get_hpm_network_ode_fn(
    mesh_per_dim: int,
    cosmo: Cosmology,
    gravity_model=None,
    pressure_model=None,
    fourier_model=None,
    gas_architecture: str = "mlp",
    precomputed_edges=None,
    integrator_type: str = "diffrax",
    training: bool = False,
):
    def hpm_ode(scale, state, kwargs):
        if len(state) == 2:
            dm_pos, dm_vel = state
            gas_pos, gas_vel, gas_latent = None, None, None
            print("dark matter only")
        elif len(state) == 4:
            dm_pos, dm_vel, gas_pos, gas_vel = state
            gas_latent = None
            print("dark matter and gas")
        elif len(state) == 5:
            dm_pos, dm_vel, gas_pos, gas_vel, gas_latent = state
            print("dark matter, gas and latent")
        else:
            raise ValueError(f"Unknown state shape {state.shape}")

        if kwargs is None:
            kwargs = {}

        # dm_force, gas_force, d_gas_latent = hpm_forces_denise(
        dm_force, gas_force, d_gas_latent = hpm_forces(
            mesh_per_dim,
            cosmo,
            scale,
            dm_pos,
            gas_pos,
            # gravity
            gravity_model=gravity_model,
            # pressure
            pressure_model=pressure_model,
            gas_vel=gas_vel,
            gas_latent=gas_latent,
            gas_architecture=gas_architecture,
            graph_edges=precomputed_edges,
            training=training,
            # fourier_model=fourier_model,
            **kwargs,
        )

        # update the positions (drift)
        drift_fac = 1.0 / (scale**3 * jnp.sqrt(jc.background.Esqr(cosmo, scale)))
        d_dm_pos = drift_fac * dm_vel

        # update the velocities (kick)
        kick_fac = 1.0 / (scale**2 * jnp.sqrt(jc.background.Esqr(cosmo, scale)))
        d_dm_vel = kick_fac * dm_force

        if gas_pos is not None:
            d_gas_pos = drift_fac * gas_vel
            d_gas_vel = kick_fac * gas_force

        # TODO
        # # the two particle species have different masses
        # d_dm_vel /= cosmo.Omega_c / (cosmo.Omega_c + cosmo.Omega_b)
        # d_gas_vel /= cosmo.Omega_b / (cosmo.Omega_c + cosmo.Omega_b)

        if len(state) == 2:
            return d_dm_pos, d_dm_vel
        elif len(state) == 4:
            return d_dm_pos, d_dm_vel, d_gas_pos, d_gas_vel
        elif len(state) == 5:
            return d_dm_pos, d_dm_vel, d_gas_pos, d_gas_vel, d_gas_latent

    if integrator_type == "odeint":
        ode_fn = lambda state, scale, args: hpm_ode(scale, state, args)
    elif integrator_type == "diffrax":
        ode_fn = lambda scale, state, args: hpm_ode(scale, state, args)
    else:
        raise ValueError(f"Unknown integrator type {integrator_type}")

    return ode_fn
