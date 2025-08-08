def hpm_ode(scale, state, kwargs):
    dm_pos, dm_vel, gas_pos, gas_vel = state

    dm_force, gas_force, gas_latent = hpm_forces(...)

    # like (3) in https://arxiv.org/pdf/2207.05509
    dm_force *= 1.5 * cosmo.Omega_m
    gas_force *= 1.5 * cosmo.Omega_m

    # update the positions (drift)
    pos_fac = 1.0 / (scale**3 * jnp.sqrt(jc.background.Esqr(cosmo, scale)))
    d_dm_pos = pos_fac * dm_vel
    d_gas_pos = pos_fac * gas_vel

    # update the velocities (kick)
    vel_fac = 1.0 / (scale**2 * jnp.sqrt(jc.background.Esqr(cosmo, scale)))
    d_dm_vel = vel_fac * dm_force
    d_gas_vel = vel_fac * gas_force

    # the two particle species have different masses
    d_dm_vel /= cosmo.Omega_c / (cosmo.Omega_c + cosmo.Omega_b)
    d_gas_vel /= cosmo.Omega_b / (cosmo.Omega_c + cosmo.Omega_b)

    return d_dm_pos, d_dm_vel, d_gas_pos, d_gas_vel


rho = cic_paint(jnp.zeros(shape=mesh_shape), gas_pos)
delta_k = jnp.fft.fftn(rho)

kvec = fftk(mesh_shape, symmetric=False)
kk = jnp.sqrt(sum((ki / jnp.pi) ** 2 for ki in kvec))
kk = jnp.where(kk == 0, 1.0, kk)

# Compute the tidal field at the position of each particle
T_xx = cic_read(jnp.fft.ifftn(-(kvec[0] ** 2) * delta_k / kk).real, gas_pos)
T_yy = cic_read(jnp.fft.ifftn(-(kvec[1] ** 2) * delta_k / kk).real, gas_pos)
T_zz = cic_read(jnp.fft.ifftn(-(kvec[2] ** 2) * delta_k / kk).real, gas_pos)
T_xy = cic_read(jnp.fft.ifftn(-(kvec[0] * kvec[1]) * delta_k / kk).real, gas_pos)
T_xz = cic_read(jnp.fft.ifftn(-(kvec[0] * kvec[2]) * delta_k / kk).real, gas_pos)
T_yz = cic_read(jnp.fft.ifftn(-(kvec[1] * kvec[2]) * delta_k / kk).real, gas_pos)

# shape (N,3,3)
T = jnp.stack(
    [jnp.stack([T_xx, T_xy, T_xz], -1), jnp.stack([T_xy, T_yy, T_yz], -1), jnp.stack([T_xz, T_yz, T_zz], -1)], -2
)
