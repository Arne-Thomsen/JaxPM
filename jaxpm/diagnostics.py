import jax
import jax.numpy as jnp

import diffrax
from diffrax import diffeqsolve, ODETerm, LeapfrogMidpoint, PIDController, SaveAt, ConstantStepSize

from jaxpm import hpm, plotting


def run_simulations(
    camels_dict,
    mesh_per_dim,
    gravity_model=None,
    pressure_model=None,
    gas_architecture=None,
    dt0=0.01,
    plot_dm=False,
    plot_gas=True,
    i_snapshots=None,
):
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

    og_ode = hpm.get_hpm_network_ode_fn(mesh_per_dim, cosmo)
    og_res = diffeqsolve(
        terms=ODETerm(og_ode),
        solver=LeapfrogMidpoint(),
        t0=scales[0],
        t1=scales[-1],
        dt0=dt0,
        y0=(dm_poss[0], dm_vels[0], gas_poss[0], gas_vels[0]),
        saveat=SaveAt(ts=scales),
        max_steps=100,
        stepsize_controller=ConstantStepSize(),
    )
    og_dm_poss, og_dm_vels, og_gas_poss, og_gas_vels = og_res.ys

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
        y0=(dm_poss[0], dm_vels[0], gas_poss[0], gas_vels[0]),
        saveat=SaveAt(ts=scales),
        max_steps=100,
        stepsize_controller=ConstantStepSize(),
    )
    nn_dm_poss, nn_dm_vels, nn_gas_poss, nn_gas_vels = nn_res.ys

    with jax.default_device(jax.devices("cpu")[0]):
        if plot_dm:
            plotting.compare_particle_evolution(
                [mesh_per_dim] * 3,
                scales,
                jnp.stack([dm_poss, og_dm_poss, nn_dm_poss], axis=0),
                title="dark matter",
                col_titles=["CAMELS", "gravity", "gravity + pressure"],
                include_pk=True,
                include_reference=True,
            )

        if plot_gas:
            plotting.compare_particle_evolution(
                [mesh_per_dim] * 3,
                scales,
                jnp.stack([gas_poss, og_gas_poss, nn_gas_poss], axis=0),
                title="gas",
                col_titles=["CAMELS", "gravity", "gravity + pressure"],
                include_pk=True,
                include_reference=True,
            )
