import jax
import jax.numpy as jnp
import jax_cosmo as jc
from jax_cosmo import Cosmology

from jaxpm.kernels import fftk, gradient_kernel, invlaplace_kernel, invnabla_kernel, longrange_kernel, gaussian_kernel
from jaxpm.painting import cic_paint, cic_read
from jaxpm.graph import get_graph_given_edges, get_graph_from_features
from jaxpm.data import get_hpm_inputs
from jaxpm.nn import MLP, ConditionedCNN


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
    pressure_architecture=None,
    graph_edges=None,
    graph_kwargs={},
    training=False,
):
    mesh_shape = [mesh_per_dim] * 3

    if pressure_architecture is None:
        if isinstance(pressure_model, MLP):
            pressure_architecture = "mlp"
        elif isinstance(pressure_model, ConditionedCNN):
            pressure_architecture = "cnn"

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
        if pressure_architecture in ["mlp", "mlp+cnn", "gnn", "offline"]:
            if pressure_architecture == "mlp":
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
            elif pressure_architecture == "mlp+cnn":
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
            elif pressure_architecture == "gnn":
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
            elif pressure_architecture == "offline":
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
                raise ValueError(f"Unknown model type {pressure_architecture}")

            if gas_latent is None:
                print("No latent variable")
                # gas_U = jnp.exp(jnp.squeeze(gas_preds))
                # gas_P = gas_U * gas_rho
                gas_P = 10 ** jnp.squeeze(gas_preds)
            else:
                print(f"With latent variable")
                # gas_U, d_gas_latent = jnp.exp(gas_preds[:, 0]), jnp.sinh(gas_preds[:, 1:])
                gas_U, d_gas_latent = jnp.exp(gas_preds[:, 0]), gas_preds[:, 1:]
                gas_P = gas_U * gas_rho

            P_gas = cic_paint(jnp.zeros(mesh_shape), gas_pos, weight=gas_P / gas_N)

        # field-level network output
        else:
            if pressure_architecture == "cnn":

                if gas_latent is not None:
                    latent_gas = gas_latent
                else:
                    latent_gas = None

                _, field_inputs = get_hpm_inputs(
                    scale,
                    gas_pos,
                    gas_vel,
                    gas_rho,
                    rho_gas,
                    gas_N,
                    mesh_shape,
                    latent_gas=latent_gas,
                    return_field=True,
                )
                preds_gas = pressure_model(field_inputs, jnp.atleast_1d(scale), training=training)
            else:
                raise ValueError(f"Unknown model type {pressure_architecture}")

            if gas_latent is None:
                print("No latent variable")
                U_gas = jnp.exp(jnp.squeeze(preds_gas))
                P_gas = U_gas * rho_gas
            else:
                print(f"With latent variable")
                # U_gas, d_gas_latent = jnp.exp(preds_gas[..., 0]), jnp.sinh(preds_gas[..., 1:])
                U_gas, d_gas_latent = jnp.exp(preds_gas[..., 0]), preds_gas[..., 1:]
                P_gas = U_gas * rho_gas

        d_gas_latent -= jnp.mean(d_gas_latent)
        P_gas_k = jnp.fft.rfftn(P_gas)

        if hasattr(pressure_model, "k_smooth"):
            print("Applying learned Gaussian kernel to P_gas_k")
            P_gas_k *= gaussian_kernel(kvec, pressure_model.k_smooth)
        if hasattr(pressure_model, "k_model"):
            print("Applying learned Fourier filter to P_gas_k")
            k = jnp.sqrt(sum((ki / jnp.pi) ** 2 for ki in kvec))
            P_gas_k += P_gas_k * pressure_model.k_model(k, jnp.atleast_1d(scale))

        def pressure(pos):
            nabla_P = jnp.stack(
                [cic_read(jnp.fft.irfftn(gradient_kernel(kvec, i) * P_gas_k), pos) for i in range(len(kvec))],
                axis=-1,
            )
            return nabla_P / jnp.maximum(jnp.expand_dims(gas_rho, axis=-1), 1e-3)

        gas_force -= pressure(gas_pos)

    return dm_force, gas_force, d_gas_latent


def get_hpm_network_ode_fn(
    mesh_per_dim: int,
    cosmo: Cosmology,
    gravity_model=None,
    pressure_model=None,
    pressure_architecture: str = None,
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

        dm_force, gas_force, d_gas_latent = hpm_forces(
            mesh_per_dim,
            cosmo,
            scale,
            dm_pos,
            gas_pos,
            # gravity
            gravity_model=gravity_model,
            # pressure
            gas_vel=gas_vel,
            gas_latent=gas_latent,
            pressure_model=pressure_model,
            pressure_architecture=pressure_architecture,
            graph_edges=precomputed_edges,
            training=training,
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

        if len(state) == 2:
            dy = d_dm_pos, d_dm_vel
        elif len(state) == 4:
            dy = d_dm_pos, d_dm_vel, d_gas_pos, d_gas_vel
        elif len(state) == 5:
            dy = d_dm_pos, d_dm_vel, d_gas_pos, d_gas_vel, d_gas_latent

        return dy

    if integrator_type == "odeint":
        ode_fn = lambda state, scale, args: hpm_ode(scale, state, args)
    elif integrator_type == "diffrax":
        ode_fn = lambda scale, state, args: hpm_ode(scale, state, args)
    else:
        raise ValueError(f"Unknown integrator type {integrator_type}")

    return ode_fn
