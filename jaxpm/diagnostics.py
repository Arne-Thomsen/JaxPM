import jax
import jax.numpy as jnp

from jaxpm import plotting, nn, training
from jaxpm.painting import cic_paint, cic_read


def run_simulations(
    camels_dict,
    mesh_per_dim,
    gravity_model=None,
    pressure_model=None,
    with_latent=False,
    i_init=0,
    i_plot=None,
    dt0=None,
    nt=None,
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

    if i_plot is None:
        i_plot = jnp.arange(i_init, scales.shape[0])

    y0 = (dm_poss[i_init], dm_vels[i_init], gas_poss[i_init], gas_vels[i_init])
    if with_latent:
        if isinstance(pressure_model, nn.MLP):
            latent_init = jnp.ones((dm_poss.shape[1], 1))
        elif isinstance(pressure_model, nn.ConditionedCNN):
            latent_init = jnp.ones(mesh_shape + [1])
        y0 += (latent_init,)

    t0 = scales[i_init]
    tsave = scales[i_plot]
    tstep = scales[i_init:]
    solve_ode = training.get_ode_solver(mesh_per_dim, cosmo)

    pm_res = solve_ode(
        y0, t0, tsave, gravity_model=None, pressure_model=None, training=False, nt=nt, dt0=dt0, tstep=tstep
    )
    pm_dm_poss, pm_dm_vels, pm_gas_poss, pm_gas_vels = pm_res[:4]
    if with_latent:
        pm_latents = pm_res[4]

    nn_res = solve_ode(y0, t0, tsave, gravity_model, pressure_model, training=False, nt=nt, dt0=dt0, tstep=tstep)
    nn_dm_poss, nn_dm_vels, nn_gas_poss, nn_gas_vels = nn_res[:4]
    if with_latent:
        nn_gas_latents = nn_res[4]

    scales = scales[i_plot]
    dm_poss = dm_poss[i_plot]
    dm_vels = dm_vels[i_plot]
    gas_poss = gas_poss[i_plot]
    gas_vels = gas_vels[i_plot]

    if plot_latent:
        n_latent = nn_gas_latents.shape[-1]

        if isinstance(pressure_model, nn.ConditionedCNN):
            nn_gas_latents = jnp.squeeze(nn_gas_latents)

        elif isinstance(pressure_model, nn.MLP):

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
                jnp.stack([dm_poss, pm_dm_poss, nn_dm_poss], axis=0),
                title="dark matter",
                col_titles=["CAMELS", "JaxPM", "JaxPM + NN"],
                include_pk=True,
                include_reference=True,
            )

        if plot_gas:
            plotting.compare_particle_evolution(
                mesh_shape,
                scales,
                jnp.stack([gas_poss, pm_gas_poss, nn_gas_poss], axis=0),
                title="gas",
                col_titles=["CAMELS", "JaxPM", "JaxPM + NN"],
                include_pk=True,
                include_reference=True,
            )

        if plot_latent:
            if isinstance(pressure_model, nn.ConditionedCNN):
                plotting.compare_field_evolution(
                    scales,
                    jnp.stack([nn_gas_latents, nn_gas_latents], axis=0),
                    # values
                    # log=True,
                    log=False,
                    # cosmetics
                    title="latent",
                    shared_colorbar=False,
                    individual_colorbars=True,
                )

            elif isinstance(pressure_model, nn.MLP):
                plotting.compare_particle_evolution(
                    mesh_shape,
                    scales,
                    jnp.stack([nn_gas_poss for _ in range(n_latent + 1)], axis=0),
                    title="latent",
                    weights=jnp.concatenate(
                        [jnp.ones((1, nn_gas_poss.shape[0], nn_gas_poss.shape[1])), weights], axis=0
                    ),
                    col_titles=["gas_pos"] + [f"latent {i}" for i in range(n_latent)],
                    # shared_colorbar=True,
                    shared_colorbar=False,
                    individual_colorbars=True,
                    log=False,
                    arcsinh=False,
                )
