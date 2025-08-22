import os, glob, h5py, tqdm, hdf5plugin
import numpy as np
import jax
import jax.numpy as jnp

from jaxpm.painting import cic_paint, cic_read
from jaxpm.kernels import fftk, gradient_kernel, invnabla_kernel
import jax_cosmo as jc


def load_CV_snapshots(
    CV,
    mesh_per_dim,
    parts_per_dim=None,
    i_snapshots=None,
    snapshots=None,
    np_seed=7,
    return_hydro=True,
    pm_units=True,
    # simulation
    CAMELS="/cluster/work/refregier/athomsen/flatiron/CAMELS",
    CODE="SIMBA",
    force_h5=False,
):
    """
    NOTE for training of the HPM-"table" network, the gas particles don't actually need to exist in all snapshots
    """

    if isinstance(i_snapshots, int):
        i_snapshots = [i_snapshots]

    # see https://camels.readthedocs.io/en/latest/parameters.html#cosmological-parameters
    cosmo = jc.Planck15(
        Omega_c=0.3 - 0.049,
        Omega_b=0.049,
        n_s=0.9624,
        h=0.6711,
        sigma8=0.8,
    )

    if not return_hydro:
        CODE += "_DM"

    RUN = os.path.join(CODE, "CV", CV)
    SIM = os.path.join(CAMELS, "Sims", RUN)
    CAT = os.path.join(CAMELS, "FOF_Subfind", RUN)

    h5_FILE = os.path.join(
        CAMELS,
        "h5",
        RUN,
        f"parts={parts_per_dim},mesh={mesh_per_dim}{'' if i_snapshots is None else f',i={i_snapshots}'}.h5",
    )
    if not force_h5 and os.path.exists(h5_FILE):
        snapshot_dict = _h5_to_dict(h5_FILE)
    else:
        print(f"Creating {h5_FILE}")

        # list all snapshots
        SNAPSHOTS = glob.glob(os.path.join(SIM, "snapshot_???.hdf5"))
        CATALOGS = glob.glob(os.path.join(CAT, "groups_???.hdf5"))

        if return_halos := len(SNAPSHOTS) == len(CATALOGS):
            print(f"Found matching catalogs")
        else:
            print(f"No matching catalogs found, returning only the snapshots")

        CATALOGS.sort()
        SNAPSHOTS.sort()

        # subselect snapshots
        assert i_snapshots is None or snapshots is None, "Only one of i_snapshots or snapshots can be specified"
        if i_snapshots is not None:
            SNAPSHOTS = [SNAPSHOTS[i] for i in i_snapshots]
            CATALOGS = [CATALOGS[i] for i in i_snapshots] if return_halos else None
            print(f"Using snapshots {SNAPSHOTS}")
        if snapshots is not None:
            SNAPSHOTS = [s for s in SNAPSHOTS if os.path.basename(s) in snapshots]
            CATALOGS = [s for s in CATALOGS if os.path.basename(s) in snapshots] if return_halos else None

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

                    # SIMBA produces some duplicate ids
                    if len(gas_ids) != len(np.unique(gas_ids)):
                        unique, unique_counts = np.unique(gas_ids, return_counts=True)
                        unique_singles = unique[unique_counts == 1]
                        gas_ids_mask = np.isin(gas_ids, unique_singles)
                        gas_ids = gas_ids[gas_ids_mask]
                        print(f"Found {len(gas_ids_mask) - np.sum(gas_ids_mask)} duplicate gas particle IDs")

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
            "redshifts": [],
            "dm_ids": [],
            "dm_poss": [],
            "dm_vels": [],
            "dm_masss": [],
        }
        if return_halos:
            snapshot_dict.update(
                {
                    "h_poss": [],
                    "h_masss": [],
                    "h_lens": [],
                    "h_ids": [],
                }
            )
        if return_hydro:
            snapshot_dict.update(
                {
                    "gas_ids": [],
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
                dm_ids = data["PartType1/ParticleIDs"][:]

                dm_pos = data["PartType1/Coordinates"][:] / 1e3  # Mpc/h
                dm_pos *= mesh_per_dim / box_size  # rescaling positions to grid coordinates

                # see https://camels.readthedocs.io/en/latest/snapshots.html?highlight=velocity#initial-conditions
                # https://www.tng-project.org/data/docs/specifications/#parttype0
                dm_vel = data["PartType1/Velocities"][:]  # v_gadget in sqrt(a) km/s
                dm_vel *= np.sqrt(scale_factor)  # -> v_peculiar in a km/s
                dm_vel *= scale_factor  # -> v_swift in a^2 km/s
                dm_vel *= (
                    mesh_per_dim / box_size
                )  # -> pm length in a^2 km/s h/Mpc, where [mesh_per_dim] = int, [box_size] = Mpc/h
                dm_vel /= 100  # -> pm velocities (a^2 H_0)

                try:
                    dm_mass_msun = data["PartType1/Masses"][:] * 1e10  # Msun/h
                    assert len(np.unique(dm_mass_msun)) == 1
                    dm_mass_msun = dm_mass_msun[0]
                except KeyError:
                    dm_mass_msun = data["Header"].attrs["MassTable"][1] * 1e10  # Msun/h

                if pm_units:
                    dm_mass = cosmo.Omega_c / (cosmo.Omega_c + cosmo.Omega_b) if return_hydro else 1.0
                else:
                    dm_mass = dm_mass_msun

                if subsample_particles:
                    i_sort = np.argsort(dm_ids)
                    dm_ids = dm_ids[i_sort]
                    dm_pos = dm_pos[i_sort]
                    dm_vel = dm_vel[i_sort]
                    dm_ids = _subsample_ordered_particles_in_boxes(
                        dm_ids, in_particles=256, out_particles=parts_per_dim
                    )
                    dm_pos = _subsample_ordered_particles_in_boxes(
                        dm_pos, in_particles=256, out_particles=parts_per_dim
                    )
                    dm_vel = _subsample_ordered_particles_in_boxes(
                        dm_vel, in_particles=256, out_particles=parts_per_dim
                    )

                snapshot_dict["dm_ids"].append(dm_ids)
                snapshot_dict["dm_poss"].append(dm_pos)
                snapshot_dict["dm_vels"].append(dm_vel)
                snapshot_dict["dm_masss"].append(np.full(dm_pos.shape[0], dm_mass))

                # gas #####################################################################################################
                if return_hydro:
                    gas_ids = data["PartType0/ParticleIDs"][:]

                    gas_pos = data["PartType0/Coordinates"][:] / 1e3  # Mpc/h
                    gas_pos *= mesh_per_dim / box_size  # rescaling positions to grid coordinates pm_len

                    gas_vel = data["PartType0/Velocities"][:]  # v_gadget in sqrt(a) km/s like for dark matter
                    # Gadget factor https://camels.readthedocs.io/en/latest/snapshots.html?highlight=velocity#initial-conditions
                    gas_vel *= np.sqrt(scale_factor)
                    gas_vel *= (
                        mesh_per_dim * scale_factor / (box_size * 100)
                    )  # pm_vel (scale for peculiar, 100 for Hubble

                    gas_mass = data["PartType0/Masses"][:] * 1e10  # Msun/h
                    if pm_units:
                        gas_mass /= dm_mass_msun + np.mean(
                            gas_mass
                        )  # [dm_mass] per particle like ~ Omega_b / (Omega_c + Omega_b)

                    # density (comoving)
                    rho_gas = cic_paint(np.zeros([mesh_per_dim] * 3), gas_pos, gas_mass)
                    gas_rho = cic_read(rho_gas, gas_pos)  # dm_mass/pm_len

                    # internal energy (physical)
                    gas_U = data["PartType0/InternalEnergy"][:]  # (km/s)^2
                    gas_U *= (
                        mesh_per_dim * scale_factor / (box_size * 100)
                    ) ** 2  # rescale like the velocity, pm_vel^2

                    # pressure (physical)
                    gamma = 5.0 / 3.0
                    gas_P = (gamma - 1.0) * gas_U * (gas_rho / scale_factor**3)  #  dm_mass*pm_vel^2/pm_len^3

                    # gas_P = (gamma - 1.0) * gas_U * gas_rho  #  dm_mass*pm_vel^2/pm_len^3

                    # pressure (comoving)
                    # gas_P *= scale_factor ** (3 * gamma)

                    if not pm_units:
                        gas_rho = data["PartType0/Density"][:] * 1e10 * (1e3) ** 3  # (Msun/h)/(Mpc/h)^3
                        gas_U = data["PartType0/InternalEnergy"][:]  # (km/s)^2
                        gas_P = (gamma - 1.0) * gas_U * gas_rho  #  (Msun/h)*(km/s)^2/(Mpc/h)^3

                    # temperature
                    gas_ne = data["PartType0/ElectronAbundance"][:]
                    yhelium = 0.0789
                    k_B = 1.38065e-16  # erg/K - NIST 2010
                    m_p = 1.67262178e-24  # gram  - NIST 2010
                    gas_T = gas_U * (1.0 + 4.0 * yhelium) / (1.0 + yhelium + gas_ne) * 1e10 * (2.0 / 3.0) * m_p / k_B

                    if subsample_particles:
                        i_sort = np.argsort(gas_ids)
                        # if len(gas_ids) != len(np.unique(gas_ids)):
                        #     print(f"WARNING! {SNAPSHOT} has duplicate gas particle IDs")

                        gas_subselect_mask = np.isin(gas_ids[i_sort], gas_sub_ids)

                        gas_ids = gas_ids[i_sort][gas_subselect_mask]
                        gas_pos = gas_pos[i_sort][gas_subselect_mask]
                        gas_vel = gas_vel[i_sort][gas_subselect_mask]
                        gas_mass = gas_mass[i_sort][gas_subselect_mask]
                        gas_rho = gas_rho[i_sort][gas_subselect_mask]
                        gas_U = gas_U[i_sort][gas_subselect_mask]
                        gas_P = gas_P[i_sort][gas_subselect_mask]
                        gas_T = gas_T[i_sort][gas_subselect_mask]

                        # NOTE pure randomness for debugging
                        # gas_ids = rng.choice(np.arange(len(gas_pos)), parts_per_dim**3, replace=False)
                        # gas_pos = gas_pos[gas_ids]
                        # gas_vel = gas_vel[gas_ids]
                        # gas_mass = gas_mass[gas_ids]
                        # gas_rho = gas_rho[gas_ids]
                        # gas_U = gas_U[gas_ids]
                        # gas_P = gas_P[gas_ids]
                        # gas_T = gas_T[gas_ids]

                    snapshot_dict["gas_ids"].append(gas_ids)
                    snapshot_dict["gas_poss"].append(gas_pos)
                    snapshot_dict["gas_vels"].append(gas_vel)
                    snapshot_dict["gas_masss"].append(gas_mass)
                    snapshot_dict["gas_rhos"].append(gas_rho)
                    snapshot_dict["gas_Us"].append(gas_U)
                    snapshot_dict["gas_Ps"].append(gas_P)
                    snapshot_dict["gas_Ts"].append(gas_T)

            # halos
            if return_halos:
                CATALOG = CATALOGS[i]
                with h5py.File(CATALOG, "r") as f:
                    h_pos = f["Group/GroupPos"][:] * mesh_per_dim / (1e3 * box_size)
                    h_mass = f["Group/GroupMass"][:] * 1e10
                    h_len = f["Group/GroupLen"][:]
                    h_ids = f["IDs"]["ID"][:]

                    snapshot_dict["h_poss"].append(h_pos)
                    snapshot_dict["h_masss"].append(h_mass)
                    snapshot_dict["h_lens"].append(h_len)
                    snapshot_dict["h_ids"].append(h_ids)

        # convert lists to np.arrays for compatible shapes
        for key, value in snapshot_dict.items():
            try:
                snapshot_dict[key] = np.squeeze(np.stack(value, axis=0))
            except (ValueError, TypeError):
                print(f"Could not stack {key}")

        _dict_to_h5(snapshot_dict, h5_FILE)

    snapshot_dict["cosmo"] = cosmo
    snapshot_dict["mesh_per_dim"] = mesh_per_dim

    return snapshot_dict


def _subsample_ordered_particles_in_boxes(particles, in_particles=256, out_particles=64):
    """
    It's important that the particles are ordered by index. Adapted from:
    https://github.com/DifferentiableUniverseInitiative/jaxpm-paper/blob/main/notebooks/dev/CAMELS_Fitting_PosVel.ipynb
    """

    assert in_particles % out_particles == 0

    if particles.ndim == 2:
        dims = particles.shape[1]
    elif particles.ndim == 1:
        dims = 1
        particles = particles[:, np.newaxis]

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

    return np.squeeze(particles)


def preprocess_snapshots(snapshot_dict, compute_tidal_tensor=False):
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
    rho_gas = vcic_paint_scalar(jnp.zeros(mesh_shape), gas_pos, cosmo.Omega_b / (cosmo.Omega_c + cosmo.Omega_b))
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

    if compute_tidal_tensor:
        kvec = fftk(mesh_shape, symmetric=False)
        kk = jnp.sqrt(sum((ki / jnp.pi) ** 2 for ki in kvec))
        kk = jnp.where(kk == 0, 1.0, kk)
        kk = kk[jnp.newaxis]

        delta_k = jnp.fft.fftn(rho_gas, axes=(1, 2, 3))

        # compute the tidal field at the position of each particle
        T_xx = vcic_read(jnp.fft.ifftn(-(kvec[0] ** 2) * delta_k / kk, axes=(1, 2, 3)).real, gas_pos)
        T_yy = vcic_read(jnp.fft.ifftn(-(kvec[1] ** 2) * delta_k / kk, axes=(1, 2, 3)).real, gas_pos)
        T_zz = vcic_read(jnp.fft.ifftn(-(kvec[2] ** 2) * delta_k / kk, axes=(1, 2, 3)).real, gas_pos)
        T_xy = vcic_read(jnp.fft.ifftn(-(kvec[0] * kvec[1]) * delta_k / kk, axes=(1, 2, 3)).real, gas_pos)
        T_xz = vcic_read(jnp.fft.ifftn(-(kvec[0] * kvec[2]) * delta_k / kk, axes=(1, 2, 3)).real, gas_pos)
        T_yz = vcic_read(jnp.fft.ifftn(-(kvec[1] * kvec[2]) * delta_k / kk, axes=(1, 2, 3)).real, gas_pos)

        T = jnp.stack(
            [
                jnp.stack([T_xx, T_xy, T_xz], axis=-1),
                jnp.stack([T_xy, T_yy, T_yz], axis=-1),
                jnp.stack([T_xz, T_yz, T_zz], axis=-1),
            ],
            axis=-2,
        )

        # symmetric, so eigh is fine
        gas_tidal_eigval, gas_tidal_eigvec = jnp.linalg.eigh(T)
        gas_tidal_eigvec = gas_tidal_eigvec.reshape(gas_tidal_eigvec.shape[0], gas_tidal_eigvec.shape[1], -1)

    # output
    particle_features = {}
    particle_features["gas_pos"] = gas_pos
    particle_features["gas_rho"] = gas_rho
    particle_features["gas_fscalar"] = gas_fscalar
    particle_features["gas_vel_disp"] = gas_vel_disp
    particle_features["gas_vel_div"] = gas_vel_div
    if compute_tidal_tensor:
        for i in range(gas_tidal_eigval.shape[-1]):
            particle_features[f"gas_tidal_eigval_{i}"] = gas_tidal_eigval[..., i]
        for i in range(gas_tidal_eigvec.shape[-1]):
            particle_features[f"gas_tidal_eigvec_{i}"] = gas_tidal_eigvec[..., i]

    particle_features["gas_P"] = snapshot_dict["gas_Ps"]
    particle_features["gas_U"] = snapshot_dict["gas_Us"]
    particle_features["gas_T"] = snapshot_dict["gas_Ts"]

    field_features = {}
    field_features["rho_gas"] = rho_gas
    field_features["fscalar_gas"] = fscalar_gas
    field_features["vel_disp_gas"] = vel_disp_gas
    field_features["vel_div_gas"] = vel_div_gas
    if compute_tidal_tensor:
        for i in range(gas_tidal_eigval.shape[-1]):
            field_features[f"tidal_eigval_{i}_gas"] = vcic_paint(
                jnp.zeros(mesh_shape), gas_pos, gas_tidal_eigval[..., i] / gas_N
            )

    field_features["P_gas"] = vcic_paint(jnp.zeros(mesh_shape), gas_pos, snapshot_dict["gas_Ps"] / gas_N)
    field_features["U_gas"] = vcic_paint(jnp.zeros(mesh_shape), gas_pos, snapshot_dict["gas_Us"] / gas_N)
    field_features["T_gas"] = vcic_paint(jnp.zeros(mesh_shape), gas_pos, snapshot_dict["gas_Ts"] / gas_N)

    return scales, particle_features, field_features


def _dict_to_h5(snapshot_dict, FILE):
    os.makedirs(os.path.dirname(FILE), exist_ok=True)
    with h5py.File(FILE, "w") as f:
        for key, array in snapshot_dict.items():
            try:
                f.create_dataset(key, data=array)
            # TODO halos fail
            except ValueError:
                print(f"Could not save {key} to {FILE}")
                continue

    print(f"Saved {FILE}")


def _h5_to_dict(FILE):
    snapshot_dict = {}
    with h5py.File(FILE, "r") as f:
        for key in f.keys():
            try:
                snapshot_dict[key] = f[key][:]
            # relevant when there's only a single index
            except ValueError:
                snapshot_dict[key] = f[key][()]

    print(f"Loaded {FILE}")
    return snapshot_dict
