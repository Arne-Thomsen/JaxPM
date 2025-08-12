import jax
import jax.numpy as jnp

import diffrax
from diffrax import diffeqsolve, ODETerm, LeapfrogMidpoint, PIDController, SaveAt, ConstantStepSize

from jaxpm import hpm, plotting
from jaxpm.painting import cic_paint, cic_read


def run_simulations(
    camels_dict,
    mesh_per_dim,
    gravity_model=None,
    pressure_model=None,
    gas_architecture=None,
    dt0=0.01,
    i_snapshots=None,
    plot_dm=False,
    plot_gas=True,
    plot_latent=False,
):
    mesh_shape = [mesh_per_dim] * 3

    cosmo = camels_dict["cosmo"]
    scales = camels_dict["scales"]

    dm_poss = camels_dict["dm_poss"]
    dm_vels = camels_dict["dm_vels"]

    gas_poss = camels_dict["gas_poss"]
    gas_vels = camels_dict["gas_vels"]

    if i_snapshots is not None:
        scales = scales[i_snapshots]
        dm_poss = dm_poss[i_snapshots]
        dm_vels = dm_vels[i_snapshots]
        gas_poss = gas_poss[i_snapshots]
        gas_vels = gas_vels[i_snapshots]

    if plot_latent:
        latent_init = jnp.ones((dm_poss.shape[1], 1))
        y0 = (dm_poss[0], dm_vels[0], gas_poss[0], gas_vels[0], latent_init)
    else:
        y0 = (dm_poss[0], dm_vels[0], gas_poss[0], gas_vels[0])

    og_ode = hpm.get_hpm_network_ode_fn(mesh_per_dim, cosmo)
    og_res = diffeqsolve(
        terms=ODETerm(og_ode),
        solver=LeapfrogMidpoint(),
        t0=scales[0],
        t1=scales[-1],
        dt0=dt0,
        y0=y0,
        saveat=SaveAt(ts=scales),
        max_steps=100,
        stepsize_controller=ConstantStepSize(),
    )
    og_dm_poss, og_dm_vels, og_gas_poss, og_gas_vels = og_res.ys[:4]
    if plot_latent:
        og_latents = og_res.ys[4]

    nn_ode = hpm.get_hpm_network_ode_fn(
        mesh_per_dim,
        cosmo,
        gravity_model=gravity_model,
        pressure_model=pressure_model,
        gas_architecture=gas_architecture,
    )
    nn_res = diffeqsolve(
        terms=ODETerm(nn_ode),
        solver=LeapfrogMidpoint(),
        t0=scales[0],
        t1=scales[-1],
        dt0=dt0,
        y0=y0,
        saveat=SaveAt(ts=scales),
        max_steps=100,
        stepsize_controller=ConstantStepSize(),
    )
    nn_dm_poss, nn_dm_vels, nn_gas_poss, nn_gas_vels = nn_res.ys[:4]
    if plot_latent:
        nn_gas_latents = nn_res.ys[4]

    if plot_latent:
        n_latent = nn_gas_latents.shape[-1]

        nn_N_gas = jax.vmap(cic_paint, in_axes=(None, 0))(jnp.zeros(mesh_shape), nn_gas_poss)
        nn_gas_N = jax.vmap(cic_read, in_axes=(0, 0))(nn_N_gas, nn_gas_poss)
        nn_gas_latents_norm = jnp.where(
            nn_gas_N[..., jnp.newaxis] != 0, nn_gas_latents / nn_gas_N[..., jnp.newaxis], nn_gas_latents
        )

        # (n_latent, n_scales, n_parts)
        weights = jnp.transpose(nn_gas_latents_norm, (2, 0, 1))

    with jax.default_device(jax.devices("cpu")[0]):
        if plot_dm:
            plotting.compare_particle_evolution(
                mesh_shape,
                scales,
                jnp.stack([dm_poss, og_dm_poss, nn_dm_poss], axis=0),
                title="dark matter",
                col_titles=["CAMELS", "gravity", "gravity + pressure"],
                include_pk=True,
                include_reference=True,
            )

        if plot_gas:
            plotting.compare_particle_evolution(
                mesh_shape,
                scales,
                jnp.stack([gas_poss, og_gas_poss, nn_gas_poss], axis=0),
                title="gas",
                col_titles=["CAMELS", "gravity", "gravity + pressure"],
                include_pk=True,
                include_reference=True,
            )

        if plot_latent:
            plotting.compare_particle_evolution(
                mesh_shape,
                scales,
                jnp.stack([nn_gas_poss for _ in range(n_latent + 1)], axis=0),
                title="latent",
                weights=jnp.concatenate([jnp.ones((1, nn_gas_poss.shape[0], nn_gas_poss.shape[1])), weights], axis=0),
                col_titles=["gas_pos"] + [f"latent {i}" for i in range(n_latent)],
                shared_colorbar=True,
                log=False,
                arcsinh=True,
            )
