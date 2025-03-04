import os, glob, h5py, tqdm, hdf5plugin
import numpy as np
import jax
import jax.numpy as jnp

from jaxpm.painting import cic_paint, cic_read
from jaxpm.kernels import fftk, gradient_kernel, invnabla_kernel
import jax_cosmo as jc


def load_CV_snapshots(
    CV_SIM, mesh_per_dim, parts_per_dim=None, i_snapshots=None, np_seed=7, return_hydro=True, pm_units=True
):
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
    # print(f"Found snapshots {SNAPSHOTS}")

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

            print(
                f"There are {len(gas_ids_intersect)} ({100*len(gas_ids_intersect)/256**3:.2f}%) gas particles that"
                f" exist in all snapshots"
            )
            rng = np.random.default_rng(np_seed)
            gas_sub_ids = rng.choice(gas_ids_intersect, parts_per_dim**3, replace=False)
    else:
        print(f"Using all particles")

    snapshot_dict = {
        "scales": [],
        "dm_poss": [],
        "dm_vels": [],
        "dm_masss": [],
    }
    if return_hydro:
        snapshot_dict.update(
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
                snapshot_dict["masses"] = masses

            redshift = data["Header"].attrs["Redshift"]
            scale_factor = data["Header"].attrs["Time"]

            snapshot_dict["scales"].append(scale_factor)

            # dark matter #############################################################################################
            dm_pos = data["PartType1/Coordinates"][:] / 1e3  # Mpc/h
            dm_pos *= mesh_per_dim / box_size  # rescaling positions to grid coordinates

            dm_vel = data["PartType1/Velocities"][:]  # peculiar velocities in km/s
            dm_vel *= mesh_per_dim * scale_factor / (box_size * 100)
            # NOTE this mysterious factor seems to be included in readgadget.read_block
            dm_vel *= np.sqrt(scale_factor)

            try:
                dm_mass_msun = data["PartType1/Masses"][:] * 1e10  # Msun/h
                assert len(jnp.unique(dm_mass_msun)) == 1
                dm_mass_msun = dm_mass_msun[0]
            except KeyError:
                dm_mass_msun = data["Header"].attrs["MassTable"][1] * 1e10  # Msun/h

            if pm_units:
                dm_mass = cosmo.Omega_c / (cosmo.Omega_c + cosmo.Omega_b) if return_hydro else 1.0
            else:
                dm_mass = dm_mass_msun

            if subsample_particles:
                dm_ids = np.argsort(data["PartType1/ParticleIDs"][:])
                dm_pos = dm_pos[dm_ids]
                dm_vel = dm_vel[dm_ids]
                dm_pos = _subsample_ordered_particles_in_boxes(dm_pos, in_particles=256, out_particles=parts_per_dim)
                dm_vel = _subsample_ordered_particles_in_boxes(dm_vel, in_particles=256, out_particles=parts_per_dim)

            snapshot_dict["dm_poss"].append(dm_pos)
            snapshot_dict["dm_vels"].append(dm_vel)
            snapshot_dict["dm_masss"].append(jnp.full(dm_pos.shape[0], dm_mass))

            # gas #####################################################################################################
            if return_hydro:
                gas_pos = data["PartType0/Coordinates"][:] / 1e3  # Mpc/h
                gas_pos *= mesh_per_dim / box_size  # rescaling positions to grid coordinates pm_len

                gas_vel = data["PartType0/Velocities"][:]  # peculiar velocities in km/s
                gas_vel *= mesh_per_dim * scale_factor / (box_size * 100)  # pm_vel (scale for peculiar, 100 for Hubble
                # NOTE this mysterious factor seems to be included in readgadget.read_block
                gas_vel *= np.sqrt(scale_factor)

                gas_mass = data["PartType0/Masses"][:] * 1e10  # Msun/h
                if pm_units:
                    gas_mass /= dm_mass_msun + jnp.mean(
                        gas_mass
                    )  # [dm_mass] per particle like ~ Omega_b / (Omega_c + Omega_b)

                # density
                rho_gas = cic_paint(jnp.zeros([mesh_per_dim] * 3), gas_pos, gas_mass)
                gas_rho = cic_read(rho_gas, gas_pos)  # dm_mass/(Mpc/h)^3
                gas_rho *= (mesh_per_dim / box_size) ** 3  # dm_mass/pm_len

                # pressure
                gas_U = data["PartType0/InternalEnergy"][:]  # (km/s)^2
                gas_U *= (mesh_per_dim * scale_factor / (box_size * 100)) ** 2  # rescale like the velocity, pm_vel^2
                # NOTE same mysterious factor as for the velocity
                gas_U *= scale_factor

                gamma = 5.0 / 3.0
                gas_P = (gamma - 1.0) * gas_U * gas_rho  #  dm_mass*pm_vel^2/dm_pos^3

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

                    # NOTE pure randomness for debugging
                    # gas_ids = rng.choice(np.arange(len(gas_pos)), parts_per_dim**3, replace=False)
                    # gas_pos = gas_pos[gas_ids]
                    # gas_vel = gas_vel[gas_ids]
                    # gas_mass = gas_mass[gas_ids]
                    # gas_rho = gas_rho[gas_ids]
                    # gas_U = gas_U[gas_ids]
                    # gas_P = gas_P[gas_ids]
                    # gas_T = gas_T[gas_ids]

                snapshot_dict["gas_poss"].append(gas_pos)
                snapshot_dict["gas_vels"].append(gas_vel)
                snapshot_dict["gas_masss"].append(gas_mass)
                snapshot_dict["gas_rhos"].append(gas_rho)
                snapshot_dict["gas_Us"].append(gas_U)
                snapshot_dict["gas_Ps"].append(gas_P)
                snapshot_dict["gas_Ts"].append(gas_T)

    snapshot_dict["cosmo"] = cosmo
    snapshot_dict["mesh_per_dim"] = mesh_per_dim

    # convert lists to jnp.arrays for compatible shapes
    for key, value in snapshot_dict.items():
        try:
            snapshot_dict[key] = jnp.squeeze(jnp.stack(value, axis=0))
        except (ValueError, TypeError):
            pass

    return snapshot_dict


def _subsample_ordered_particles_in_boxes(particles, in_particles=256, out_particles=64):
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


def preprocess_snapshots(snapshot_dict):
    mesh_shape = [snapshot_dict["mesh_per_dim"]] * 3

    # vmap over the snapshots
    vcic_paint_scalar = jax.vmap(cic_paint, in_axes=(None, 0, None))
    vcic_paint = jax.vmap(cic_paint, in_axes=(None, 0, 0))
    vcic_read = jax.vmap(cic_read, in_axes=(0, 0))

    # vmap over features (like velocity components)
    vvcic_paint = jax.vmap(vcic_paint, in_axes=(None, None, -1), out_axes=-1)
    vvcic_read = jax.vmap(vcic_read, in_axes=(-1, None), out_axes=-1)

    cosmo = snapshot_dict["cosmo"]
    scales = snapshot_dict["scales"]

    gas_pos = snapshot_dict["gas_poss"]
    gas_vel = snapshot_dict["gas_vels"]

    # rho
    rho_gas = vcic_paint_scalar(jnp.zeros(mesh_shape), gas_pos, cosmo.Omega_b / cosmo.Omega_c)
    gas_rho = vcic_read(rho_gas, gas_pos)

    # fscalar
    kvec = fftk(mesh_shape)
    delta_k = jnp.fft.rfftn(rho_gas, axes=(1, 2, 3))
    fscalar_gas = jnp.fft.irfftn(delta_k * invnabla_kernel(kvec), axes=(1, 2, 3))
    gas_fscalar = vcic_read(fscalar_gas, gas_pos)

    # velocity dispersion
    N_gas = vcic_paint_scalar(jnp.zeros(mesh_shape), gas_pos, 1)
    gas_N = vcic_read(N_gas, gas_pos)

    vel_mean_gas = vvcic_paint(jnp.zeros(mesh_shape), gas_pos, gas_vel / gas_N[..., jnp.newaxis])
    gas_vel_mean = vvcic_read(vel_mean_gas, gas_pos)
    gas_vel_disp = jnp.sum((gas_vel_mean - gas_vel) ** 2, axis=-1)
    vel_disp_gas = vcic_paint(jnp.zeros(mesh_shape), gas_pos, gas_vel_disp / gas_N)

    # velocity divergence
    vel_gas_k = jnp.fft.rfftn(vel_mean_gas, axes=(1, 2, 3))
    gas_vel_div = [
        vcic_read(jnp.fft.irfftn(gradient_kernel(kvec, i) * vel_gas_k[..., i], axes=(1, 2, 3)), gas_pos)
        for i in range(len(kvec))
    ]
    gas_vel_div = jnp.stack(gas_vel_div, axis=-1)
    gas_vel_div = jnp.sum(gas_vel_div, axis=-1)
    vel_div_gas = vcic_paint(jnp.zeros(mesh_shape), gas_pos, gas_vel_div / gas_N)

    # output
    particle_features = {}
    particle_features["gas_pos"] = gas_pos
    particle_features["gas_rho"] = gas_rho
    particle_features["gas_fscalar"] = gas_fscalar
    particle_features["gas_vel_disp"] = gas_vel_disp
    particle_features["gas_vel_div"] = gas_vel_div

    particle_features["gas_P"] = snapshot_dict["gas_Ps"]
    particle_features["gas_U"] = snapshot_dict["gas_Us"]
    particle_features["gas_T"] = snapshot_dict["gas_Ts"]

    field_features = {}
    field_features["rho_gas"] = rho_gas
    field_features["fscalar_gas"] = fscalar_gas
    field_features["vel_disp_gas"] = vel_disp_gas
    field_features["vel_div_gas"] = vel_div_gas

    field_features["P_gas"] = vcic_paint(jnp.zeros(mesh_shape), gas_pos, snapshot_dict["gas_Ps"] / gas_N)
    field_features["U_gas"] = vcic_paint(jnp.zeros(mesh_shape), gas_pos, snapshot_dict["gas_Us"] / gas_N)
    field_features["T_gas"] = vcic_paint(jnp.zeros(mesh_shape), gas_pos, snapshot_dict["gas_Ts"] / gas_N)

    return scales, particle_features, field_features
