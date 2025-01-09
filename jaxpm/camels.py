import os, glob, h5py, tqdm, hdf5plugin
import numpy as np
import jax.numpy as jnp

from jaxpm.painting import cic_paint, cic_read
import jax_cosmo as jc


def load_CV_snapshots(CV_SIM, mesh_per_dim, parts_per_dim=None, i_snapshots=None, np_seed=7, return_hydro=True):
    """
    NOTE for training of the HPM-"table" network, the gas particles don't actually need to exist in all snapshots
    """

    # see https://camels.readthedocs.io/en/latest/parameters.html#cosmological-parameters
    cosmo = jc.Planck15(
        Omega_c=0.3 - 0.049,
        Omega_b=0.049,
        n_s=0.9624,
        h=0.6711,
        sigma8=0.8,
    )

    # list all snapshots
    SNAPSHOTS = glob.glob(os.path.join(CV_SIM, "snapshot_???.hdf5"))
    SNAPSHOTS.sort()

    if i_snapshots is not None:
        SNAPSHOTS = [SNAPSHOTS[i] for i in i_snapshots]
        print(f"Using snapshots {SNAPSHOTS}")

    subsample_particles = parts_per_dim is not None
    if subsample_particles:
        print(f"Selecting {parts_per_dim**3} dark matter (deterministic)")

        if return_hydro:
            print(f"Selecting {parts_per_dim**3} gas particles (random)")

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

            rng = np.random.default_rng(np_seed)
            gas_sub_ids = rng.choice(gas_ids_intersect, parts_per_dim**3, replace=False)
    else:
        print(f"Using all particles")

    out_dict = {
        "scales": [],
        "dm_poss": [],
        "dm_vels": [],
    }
    if return_hydro:
        out_dict.update(
            {
                "gas_poss": [],
                "gas_vels": [],
                "gas_masss": [],
                "gas_rhos": [],
                "gas_Us": [],
                "gas_Ts": [],
                "gas_Ps": [],
            }
        )

    for i, SNAPSHOT in tqdm.tqdm(enumerate(SNAPSHOTS), total=len(SNAPSHOTS), desc="loading snapshots"):
        with h5py.File(SNAPSHOT, "r") as data:
            # constants ###############################################################################################
            if i == 0:
                box_size = data["Header"].attrs["BoxSize"] / 1e3  # size of the snapshot in comoving Mpc/h

                # h = data["Header"].attrs["HubbleParam"]  # value of the hubble parameter in 100 km/s/(Mpc/h)
                # Omega_m = data["Header"].attrs["Omega0"]
                # Omega_L = data["Header"].attrs["OmegaLambda"]
                # Omega_b = data["Header"].attrs["OmegaBaryon"]
                masses = data["Header"].attrs["MassTable"] * 1e10  # masses of the particles in Msun/h
                out_dict["masses"] = masses

            redshift = data["Header"].attrs["Redshift"]
            scale_factor = data["Header"].attrs["Time"]

            out_dict["scales"].append(scale_factor)

            # dark matter #############################################################################################
            dm_pos = data["PartType1/Coordinates"][:] / 1e3  # Mpc/h
            dm_pos *= mesh_per_dim / box_size  # rescaling positions to grid coordinates

            dm_vel = data["PartType1/Velocities"][:]  # peculiar velocities in km/s
            dm_vel *= mesh_per_dim * scale_factor / (box_size * 100)
            # NOTE this mysterious factor seems to be included in readgadget.read_block
            dm_vel *= np.sqrt(scale_factor)

            if subsample_particles:
                dm_ids = np.argsort(data["PartType1/ParticleIDs"][:])
                dm_pos = dm_pos[dm_ids]
                dm_vel = dm_vel[dm_ids]
                dm_pos = subsample_ordered_particles_in_boxes(dm_pos, in_particles=256, out_particles=parts_per_dim)
                dm_vel = subsample_ordered_particles_in_boxes(dm_vel, in_particles=256, out_particles=parts_per_dim)

            out_dict["dm_poss"].append(dm_pos)
            out_dict["dm_vels"].append(dm_vel)

            # gas #####################################################################################################
            if return_hydro:
                gas_pos = data["PartType0/Coordinates"][:] / 1e3  # Mpc/h
                gas_pos *= mesh_per_dim / box_size  # rescaling positions to grid coordinates

                gas_vel = data["PartType0/Velocities"][:]  # peculiar velocities in km/s
                gas_vel *= mesh_per_dim * scale_factor / (box_size * 100)  # scale for peculiar, 100 for Hubble
                # NOTE this mysterious factor seems to be included in readgadget.read_block
                gas_vel *= np.sqrt(scale_factor)

                gas_mass = data["PartType0/Masses"][:] * 1e10  # Msun/h

                # density
                rho_gas = cic_paint(jnp.zeros([mesh_per_dim] * 3), gas_pos, gas_mass)
                gas_rho = cic_read(rho_gas, gas_pos)
                gas_rho *= (mesh_per_dim / box_size) ** 3  # (Msun/h)/(Mpc/h)^3

                # pressure
                gas_U = data["PartType0/InternalEnergy"][:]  # (km/s)^2
                gas_U *= (mesh_per_dim * scale_factor / (box_size * 100)) ** 2  # rescale like the velocity
                gas_U *= scale_factor

                gamma = 5.0 / 3.0
                P_gas = cic_paint(
                    jnp.zeros([mesh_per_dim] * 3), gas_pos, (gamma - 1.0) * gas_U * cosmo.Omega_b / cosmo.Omega_c
                )  # dark matter particle mass units, not Msun/h
                # P_gas = cic_paint(jnp.zeros([mesh_per_dim] * 3), gas_pos, (gamma - 1.0) * gas_U * gas_mass)
                gas_P = cic_read(P_gas, gas_pos)
                gas_P *= (mesh_per_dim / box_size) ** 3  #  dm_mass*vel^2/pos^3

                # directly from CAMELS
                # gas_rho = data["PartType0/Density"][:] * 1e10 * (1e3) ** 3  # (Msun/h)/(Mpc/h)^3
                # gas_P = (gamma - 1.0) * gas_U * gas_rho  #  (Msun/h)*(km/s)^2/(Mpc/h)^3

                # temperature
                gas_ne = data["PartType0/ElectronAbundance"][:]
                yhelium = 0.0789
                k_B = 1.38065e-16  # erg/K - NIST 2010
                m_p = 1.67262178e-24  # gram  - NIST 2010
                gas_T = gas_U * (1.0 + 4.0 * yhelium) / (1.0 + yhelium + gas_ne) * 1e10 * (2.0 / 3.0) * m_p / k_B

                if subsample_particles:
                    gas_ids = np.argsort(data["PartType0/ParticleIDs"][:])
                    gas_mask = np.isin(data["PartType0/ParticleIDs"][:][gas_ids], gas_sub_ids)

                    gas_pos = gas_pos[gas_ids][gas_mask]
                    gas_vel = gas_vel[gas_ids][gas_mask]
                    gas_mass = gas_mass[gas_ids][gas_mask]
                    gas_rho = gas_rho[gas_ids][gas_mask]
                    gas_U = gas_U[gas_ids][gas_mask]
                    gas_P = gas_P[gas_ids][gas_mask]
                    gas_T = gas_T[gas_ids][gas_mask]

                out_dict["gas_poss"].append(gas_pos)
                out_dict["gas_vels"].append(gas_vel)
                out_dict["gas_masss"].append(gas_mass)
                out_dict["gas_rhos"].append(gas_rho)
                out_dict["gas_Us"].append(gas_U)
                out_dict["gas_Ps"].append(gas_P)
                out_dict["gas_Ts"].append(gas_T)

    out_dict["cosmo"] = cosmo

    # convert lists to jnp.arrays for compatible shapes
    for key, value in out_dict.items():
        try:
            out_dict[key] = jnp.squeeze(jnp.stack(value, axis=0))
        except (ValueError, TypeError):
            pass

    return out_dict


def subsample_ordered_particles_in_boxes(particles, in_particles=256, out_particles=64):
    """
    It's important that the particles are ordered by index. Adapted from:
    https://github.com/DifferentiableUniverseInitiative/jaxpm-paper/blob/main/notebooks/dev/CAMELS_Fitting_PosVel.ipynb
    """

    assert in_particles % out_particles == 0

    dims = 3
    sub_fac = in_particles // out_particles

    # divide the simulation volume into sub_fac x sub_fac x sub_fac boxes containing out_particles each
    particles = (
        particles.reshape(sub_fac, sub_fac, sub_fac, out_particles, out_particles, out_particles, dims)
        .transpose(0, 3, 1, 4, 2, 5, 6)
        .reshape(-1, dims)
    )
    # downsampling
    particles = particles.reshape([in_particles, in_particles, in_particles, dims])[
        ::sub_fac, ::sub_fac, ::sub_fac, :
    ].reshape([-1, dims])

    return particles
