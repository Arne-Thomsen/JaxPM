import os

from distributed_utils import initialize_distributed, is_on_cluster

os.environ["EQX_ON_ERROR"] = "nan"  # avoid an allgather caused by diffrax
initialize_distributed()
import jax

size = jax.device_count()

import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np
from diffrax import (ConstantStepSize, Dopri5, LeapfrogMidpoint, ODETerm,
                     SaveAt, diffeqsolve)
from jax.experimental import mesh_utils
from jax.experimental.multihost_utils import process_allgather
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from jaxpm.kernels import interpolate_power_spectrum
from jaxpm.painting import cic_paint_dx
from jaxpm.pm import linear_field, lpt, make_ode_fn

size = 64
mesh_shape = [size] * 3
box_size = [float(size)] * 3
snapshots = jnp.linspace(0.1, 1., 4)
halo_size = 4
pdims = (1, 1)
mesh = None
sharding = None
if jax.device_count() > 1:
    pdims = (2, 4)
    devices = mesh_utils.create_device_mesh(pdims)
    mesh = Mesh(devices.T, axis_names=('x', 'y'))
    sharding = NamedSharding(mesh, P('x', 'y'))


@jax.jit
def run_simulation(omega_c, sigma8):
    # Create a small function to generate the matter power spectrum
    k = jnp.logspace(-4, 1, 128)
    pk = jc.power.linear_matter_power(
        jc.Planck15(Omega_c=omega_c, sigma8=sigma8), k)

    pk_fn = lambda x: interpolate_power_spectrum(x, k, pk, sharding)
    # Create initial conditions
    initial_conditions = linear_field(mesh_shape,
                                      box_size,
                                      pk_fn,
                                      sharding=sharding,
                                      seed=jax.random.PRNGKey(0))

    cosmo = jc.Planck15(Omega_c=omega_c, sigma8=sigma8)

    # Initial displacement
    dx, p, _ = lpt(cosmo,
                   initial_conditions,
                   0.1,
                   halo_size=halo_size,
                   sharding=sharding)
    # return initial_conditions, cic_paint_dx(dx,
    #                                         halo_size=halo_size,
    #                                         sharding=sharding), None, None
    # Evolve the simulation forward
    ode_fn = make_ode_fn(mesh_shape, halo_size=halo_size, sharding=sharding)
    term = ODETerm(
        lambda t, state, args: jnp.stack(ode_fn(state, t, args), axis=0))
    solver = LeapfrogMidpoint()

    stepsize_controller = ConstantStepSize()
    res = diffeqsolve(term,
                      solver,
                      t0=0.1,
                      t1=1.,
                      dt0=0.01,
                      y0=jnp.stack([dx, p], axis=0),
                      args=cosmo,
                      saveat=SaveAt(ts=snapshots),
                      stepsize_controller=stepsize_controller)

    # Return the simulation volume at requested
    states = res.ys
    field = cic_paint_dx(dx, halo_size=halo_size, sharding=sharding)
    final_fields = [
        cic_paint_dx(state[0], halo_size=halo_size, sharding=sharding)
        for state in states
    ]

    return initial_conditions, field, final_fields, res.stats


# Run the simulation
distributed_str = "distributed" if mesh is not None else "single device"
print(f"running {distributed_str} simulation")
init, field, final_fields, stats = run_simulation(0.32, 0.8)

# # Print the statistics
print(stats)
print(f"done now saving")
if is_on_cluster():
    rank = jax.process_index()
    # # save the final state
    np.save(f'initial_conditions_{rank}.npy', init.addressable_data(0))
    np.save(f'field_{rank}.npy', field.addressable_data(0))

    if final_fields is not None:
        for i, final_field in enumerate(final_fields):
            np.save(f'final_field_{i}_{rank}.npy',
                    final_field.addressable_data(0))
else:
    gathered_init = process_allgather(init, tiled=True)
    gathered_field = process_allgather(field, tiled=True)
    np.save(f'initial_conditions.npy', gathered_init)
    np.save(f'field.npy', gathered_field)

    if final_fields is not None:
        for i, final_field in enumerate(final_fields):
            gathered_final_field = process_allgather(final_field, tiled=True)
            np.save(f'final_field_{i}.npy', gathered_final_field)

print(f"Finished!!")
