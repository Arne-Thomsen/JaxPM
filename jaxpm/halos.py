import jax.numpy as jnp
from jaxpm.painting import cic_paint


def halo_density_profile(
    particle_pos,
    halo_pos,
    particle_mass=None,
    num_bins=32,
    mesh_per_dim_input=64,
    mesh_per_dim_internal=512,
    box_size=None,
    periodic=True,
    r_max_quantile=0.99,
):
    # rescale pm coordinates
    particle_pos = particle_pos * mesh_per_dim_internal / mesh_per_dim_input
    halo_pos = halo_pos * mesh_per_dim_internal / mesh_per_dim_input

    if particle_mass is None:
        rho_field = cic_paint(jnp.zeros([mesh_per_dim_internal] * 3), particle_pos)
    else:
        rho_field = cic_paint(jnp.zeros([mesh_per_dim_internal] * 3), particle_pos, particle_mass)

    nx, ny, nz = rho_field.shape

    def pbc_distance(pos1, pos2):
        return (pos1 - pos2 + mesh_per_dim_internal / 2) % mesh_per_dim_internal - mesh_per_dim_internal / 2

    if periodic:
        x = pbc_distance(jnp.arange(nx) + 0.5, halo_pos[0])
        y = pbc_distance(jnp.arange(ny) + 0.5, halo_pos[1])
        z = pbc_distance(jnp.arange(nz) + 0.5, halo_pos[2])
    else:
        x = jnp.arange(nx) + 0.5 - halo_pos[0]
        y = jnp.arange(ny) + 0.5 - halo_pos[1]
        z = jnp.arange(nz) + 0.5 - halo_pos[2]

    xx, yy, zz = jnp.meshgrid(x, y, z, indexing="ij")
    rr = jnp.sqrt(xx**2 + yy**2 + zz**2)

    # define binning range
    r_min = rr.min()
    if periodic:
        r_dist = pbc_distance(particle_pos, halo_pos) ** 2
    else:
        r_dist = (particle_pos - halo_pos) ** 2
    r_max = jnp.quantile(jnp.sqrt(jnp.sum(r_dist, axis=-1)), r_max_quantile)

    r_bins = jnp.geomspace(r_min, r_max, num_bins + 1)
    indices = jnp.digitize(rr, r_bins) - 1

    rho_binned = []
    for i in range(num_bins):
        rho_binned.append(jnp.mean(rho_field[indices == i]))
    rho_binned = jnp.array(rho_binned)

    r_bin_centers = 0.5 * (r_bins[:-1] + r_bins[1:])

    if box_size is not None:
        print("returning radius in Mpc/h")
        r_bin_centers *= box_size / mesh_per_dim_internal
    else:
        r_bin_centers *= mesh_per_dim_input / mesh_per_dim_internal

    return r_bin_centers, rho_binned
