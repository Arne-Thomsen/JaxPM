import os, glob, h5py, tqdm
import numpy as np
import jax.numpy as jnp


def load_CV_camels_hydro(CV_SIM, mesh_per_dim, parts_per_dim=None, i_snapshots=None, np_seed=7):
    # list all snapshots
    SNAPSHOTS = glob.glob(os.path.join(CV_SIM, "snapshot_???.hdf5"))
    SNAPSHOTS.sort()
    if i_snapshots is not None:
        SNAPSHOTS = [SNAPSHOTS[i] for i in i_snapshots]
        print(f"Using snapshots {SNAPSHOTS}")

    subsample_particles = parts_per_dim is not None
    if subsample_particles:
        # only consider gas particles that exist for all snapshots
        for i, SNAPSHOT in tqdm.tqdm(
            enumerate(SNAPSHOTS), total=len(SNAPSHOTS), desc="finding unique gas particle indices"
        ):
            with h5py.File(SNAPSHOT, "r") as data:
                gas_ids = data["PartType0/ParticleIDs"][:]

            if i == 0:
                gas_ids_intersect = gas_ids
            else:
                gas_ids_intersect = np.intersect1d(gas_ids_intersect, gas_ids)

        print(f"Selecting {parts_per_dim**3} dark matter (deterministic) and gas (random) particles")
        rng = np.random.default_rng(np_seed)
        gas_sub_ids = rng.choice(gas_ids_intersect, parts_per_dim**3, replace=False)
    else:
        print(f"Using all gas and dark matter particles")

    scales = []

    dm_poss = []
    dm_vels = []

    gas_poss = []
    gas_vels = []
    gas_masss = []
    gas_Ts = []
    gas_Ps = []
    gas_rhos = []
    for i, SNAPSHOT in tqdm.tqdm(enumerate(SNAPSHOTS), total=len(SNAPSHOTS), desc="loading snapshots"):
        with h5py.File(SNAPSHOT, "r") as data:
            # constants ###############################################################################################
            if i == 0:
                box_size = data["Header"].attrs["BoxSize"] / 1e3  # size of the snapshot in comoving Mpc/h
                scale_factor = data["Header"].attrs["Time"]  # scale factor
                h = data["Header"].attrs["HubbleParam"]  # value of the hubble parameter in 100 km/s/(Mpc/h)
                masses = data["Header"].attrs["MassTable"] * 1e10  # masses of the particles in Msun/h
                Omega_m = data["Header"].attrs["Omega0"]
                Omega_L = data["Header"].attrs["OmegaLambda"]
                Omega_b = data["Header"].attrs["OmegaBaryon"]

            redshift = data["Header"].attrs["Redshift"]  # reshift of the snapshot

            scales.append((1.0 / (1 + redshift)))

            # dark matter #############################################################################################
            dm_pos = data["PartType1/Coordinates"][:] / 1e3  # Mpc/h
            dm_pos *= mesh_per_dim / box_size  # rescaling positions to grid coordinates
            dm_vel = data["PartType1/Velocities"][:]  # peculiar velocities in km/s
            dm_vel *= mesh_per_dim * (1.0 / (1 + redshift)) / (box_size * 100)

            # TODO implement grid based subsampling
            if subsample_particles:
                dm_ids = np.argsort(data["PartType1/ParticleIDs"][:] - 1)  # IDs starting from 0
                dm_pos = dm_pos[dm_ids]
                dm_vel = dm_vel[dm_ids]
                dm_pos = subsample_ordered_particles_in_boxes(dm_pos, in_particles=256, out_particles=parts_per_dim)
                dm_vel = subsample_ordered_particles_in_boxes(dm_vel, in_particles=256, out_particles=parts_per_dim)

            dm_poss.append(dm_pos)
            dm_vels.append(dm_vel)

            # gas #####################################################################################################
            gas_pos = data["PartType0/Coordinates"][:] / 1e3  # Mpc/h
            gas_pos *= mesh_per_dim / box_size  # rescaling positions to grid coordinates
            gas_vel = data["PartType0/Velocities"][:]  # peculiar velocities in km/s
            gas_vel *= mesh_per_dim * (1.0 / (1 + redshift)) / (box_size * 100)
            gas_mass = data["PartType0/Masses"][:] * 1e10  # Msun/h

            gas_rho = data["/PartType0/Density"][:] * 1e10 * (1e3) ** 3  # (Msun/h)/(Mpc/h)^3
            gas_U = data["/PartType0/InternalEnergy"][:]  # (km/s)^2
            gas_ne = data["/PartType0/ElectronAbundance"][:]

            # pressure
            gamma = 5.0 / 3.0
            gas_P = (gamma - 1.0) * gas_U * gas_rho  # units are (Msun/h)*(km/s)^2/(Mpc/h)^3

            # temperature
            yhelium = 0.0789
            k_B = 1.38065e-16  # erg/K - NIST 2010
            m_p = 1.67262178e-24  # gram  - NIST 2010
            gas_T = gas_U * (1.0 + 4.0 * yhelium) / (1.0 + yhelium + gas_ne) * 1e10 * (2.0 / 3.0) * m_p / k_B

            if subsample_particles:
                gas_mask = np.isin(data["PartType0/ParticleIDs"][:], gas_sub_ids)
                gas_pos = gas_pos[gas_mask]
                gas_vel = gas_vel[gas_mask]
                gas_mass = gas_mass[gas_mask]
                gas_rho = gas_rho[gas_mask]
                gas_P = gas_P[gas_mask]
                gas_T = gas_T[gas_mask]

            gas_poss.append(gas_pos)
            gas_vels.append(gas_vel)
            gas_masss.append(gas_mass)
            gas_rhos.append(gas_rho)
            gas_Ps.append(gas_P)
            gas_Ts.append(gas_T)

    out_dict = {
        # scalars
        "Omega_m": Omega_m,
        "Omega_b": Omega_b,
        "masses": masses,
        # lists of arrays
        "scales": scales,
        "dm_poss": dm_poss,
        "dm_vels": dm_vels,
        "gas_poss": gas_poss,
        "gas_vels": gas_vels,
        "gas_masss": gas_masss,
        "gas_rhos": gas_rhos,
        "gas_Ts": gas_Ts,
        "gas_Ps": gas_Ps,
    }

    # convert lists to jnp.arrays for compatible shapes
    for key, value in out_dict.items():
        try:
            out_dict[key] = jnp.squeeze(jnp.stack(value, axis=0))
        except (ValueError, TypeError):
            pass

    return out_dict


def subsample_ordered_particles_in_boxes(particles, in_particles=256, out_particles=64):
    """It's important that the particles are ordered by index"""

    assert in_particles % out_particles == 0

    sub_fac = in_particles // out_particles

    # divide the simulation volume into sub_fac x sub_fac x sub_fac boxes containing out_particles each
    particles = (
        particles.reshape(sub_fac, sub_fac, sub_fac, out_particles, out_particles, out_particles, 3)
        .transpose(0, 3, 1, 4, 2, 5, 6)
        .reshape(-1, 3)
    )
    # downsampling
    particles = particles.reshape([in_particles, in_particles, in_particles, 3])[
        ::sub_fac, ::sub_fac, ::sub_fac, :
    ].reshape([-1, 3])

    return particles
