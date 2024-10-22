from functools import partial

import jax
import jax.numpy as jnp
import jax_cosmo as jc
from jax.sharding import PartitionSpec as P

from jaxpm.distributed import (autoshmap, fft3d, get_local_shape, ifft3d,
                               normal_field)
from jaxpm.growth import (dGf2a, dGfa, growth_factor, growth_factor_second,
                          growth_rate, growth_rate_second)
from jaxpm.kernels import (PGD_kernel, fftk, gradient_kernel, invlaplace_kernel,
                           longrange_kernel)
from jaxpm.painting import cic_paint, cic_paint_dx, cic_read, cic_read_dx


def pm_forces(positions,
              mesh_shape=None,
              delta=None,
              r_split=0,
              halo_size=0,
              sharding=None):
    """
    Computes gravitational forces on particles using a PM scheme
    """
    if mesh_shape is None:
        assert (delta is not None),\
          "If mesh_shape is not provided, delta should be provided"
        mesh_shape = delta.shape

    if delta is None:
        field = cic_paint_dx(positions, halo_size=halo_size, sharding=sharding)
        delta_k = fft3d(field)
    elif jnp.isrealobj(delta):
        delta_k = fft3d(delta)
    else:
        delta_k = delta

    kvec = fftk(delta_k)
    # Computes gravitational potential
    pot_k = delta_k * invlaplace_kernel(kvec) * longrange_kernel(kvec,
                                                              r_split=r_split)
    # Computes gravitational forces
    forces = jnp.stack([
        cic_read_dx(ifft3d( - gradient_kernel(kvec, i) * pot_k),
                    halo_size=halo_size,
                    sharding=sharding) for i in range(3)
    ],
                       axis=-1)

    return forces


def lpt(cosmo, initial_conditions, a, halo_size=0, sharding=None,order=1):
    """
    Computes first and second order LPT displacement and momentum, 
    e.g. Eq. 2 and 3 [Jenkins2010](https://arxiv.org/pdf/0910.0258)
    """
    gpu_mesh = sharding.mesh if sharding is not None else None
    spec = sharding.spec if sharding is not None else P()
    local_mesh_shape = (*get_local_shape(initial_conditions.shape, sharding),
                        3)
    displacement = autoshmap(
      partial(jnp.zeros, shape=(local_mesh_shape), dtype='float32'),
      gpu_mesh=gpu_mesh,
      in_specs=(),
      out_specs=spec)()  # yapf: disable


    a = jnp.atleast_1d(a)
    E = jnp.sqrt(jc.background.Esqr(cosmo, a)) 
    delta_k = fft3d(initial_conditions)
    initial_force = pm_forces(displacement,
                              delta=delta_k,
                              halo_size=halo_size,
                              sharding=sharding)
    dx = growth_factor(cosmo, a) * initial_force
    p = a**2 * growth_rate(cosmo, a) * E * dx
    f = a**2 * E * dGfa(cosmo,a) * initial_force
    if order == 2:
        kvec = fftk(delta_k)
        pot_k = delta_k * invlaplace_kernel(kvec)

        delta2 = 0
        shear_acc = 0
        # for i, ki in enumerate(kvec):
        for i in range(3):
            # Add products of diagonal terms = 0 + s11*s00 + s22*(s11+s00)...
            # shear_ii = jnp.fft.irfftn(- ki**2 * pot_k)
            nabla_i_nabla_i = gradient_kernel(kvec, i)**2
            shear_ii = jnp.fft.irfftn(nabla_i_nabla_i * pot_k)
            delta2 += shear_ii * shear_acc 
            shear_acc += shear_ii

            # for kj in kvec[i+1:]:
            for j in range(i+1, 3):
                # Substract squared strict-up-triangle terms
                # delta2 -= jnp.fft.irfftn(- ki * kj * pot_k)**2
                nabla_i_nabla_j = gradient_kernel(kvec, i) * gradient_kernel(kvec, j)
                delta2 -= jnp.fft.irfftn(nabla_i_nabla_j * pot_k)**2
        
        delta_k2 = fft3d(delta2)
        init_force2 = pm_forces(displacement, delta=delta_k2,halo_size=halo_size,sharding=sharding)
        # NOTE: growth_factor_second is renormalized: - D2 = 3/7 * growth_factor_second
        dx2 = 3/7 * growth_factor_second(cosmo, a) * init_force2
        p2 = a**2 * growth_rate_second(cosmo, a) * E * dx2
        f2 = a**2 * E * dGf2a(cosmo, a) * init_force2

        dx += dx2
        p  += p2
        f  += f2

    return dx, p, f


def linear_field(mesh_shape, box_size, pk, seed, sharding=None):
    """
    Generate initial conditions.
    """
    # Initialize a random field with one slice on each gpu
    field = normal_field(mesh_shape, seed=seed, sharding=sharding)
    field = fft3d(field)
    kvec = fftk(field)
    kmesh = sum((kk / box_size[i] * mesh_shape[i])**2
                for i, kk in enumerate(kvec))**0.5
    pkmesh = pk(kmesh) * (mesh_shape[0] * mesh_shape[1] * mesh_shape[2]) / (
        box_size[0] * box_size[1] * box_size[2])

    field = field * (pkmesh)**0.5
    field = ifft3d(field)
    return field


def make_ode_fn(mesh_shape, halo_size=0, sharding=None):

    def nbody_ode(state, a, cosmo):
        """
        state is a tuple (position, velocities)
        """
        pos, vel = state

        forces = pm_forces(
            pos, mesh_shape=mesh_shape, halo_size=halo_size,
            sharding=sharding) * 1.5 * cosmo.Omega_m

        # Computes the update of position (drift)
        dpos = 1. / (a**3 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * vel

        # Computes the update of velocity (kick)
        dvel = 1. / (a**2 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * forces

        return dpos, dvel

    return nbody_ode

def get_ode_fn(cosmo, mesh_shape, halo_size=0, sharding=None):

    def nbody_ode(a, state, args):
        """
        State is an array [position, velocities]

        Compatible with [Diffrax API](https://docs.kidger.site/diffrax/)
        """
        pos, vel = state
        forces = pm_forces(pos, mesh_shape, halo_size=halo_size, sharding=sharding) * 1.5 * cosmo.Omega_m

        # Computes the update of position (drift)
        dpos = 1. / (a**3 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * vel
        
        # Computes the update of velocity (kick)
        dvel = 1. / (a**2 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * forces

        return jnp.stack([dpos, dvel])

    return nbody_ode


def pgd_correction(pos, mesh_shape, params):
    """
    improve the short-range interactions of PM-Nbody simulations with potential gradient descent method, 
    based on https://arxiv.org/abs/1804.00671

    args:
      pos: particle positions [npart, 3]
      params: [alpha, kl, ks] pgd parameters
    """
    kvec = fftk(mesh_shape)
    delta = cic_paint(jnp.zeros(mesh_shape), pos)
    alpha, kl, ks = params
    delta_k = jnp.fft.rfftn(delta)
    PGD_range=PGD_kernel(kvec, kl, ks)
    
    pot_k_pgd=(delta_k * invlaplace_kernel(kvec))*PGD_range

    forces_pgd= jnp.stack([cic_read(jnp.fft.irfftn(- gradient_kernel(kvec, i)*pot_k_pgd), pos) 
                      for i in range(3)],axis=-1)
    
    dpos_pgd = forces_pgd*alpha
   
    return dpos_pgd


def make_neural_ode_fn(model, mesh_shape):
    def neural_nbody_ode(state, a, cosmo:Cosmology, params):
        """
        state is a tuple (position, velocities)
        """
        pos, vel = state
        kvec = fftk(mesh_shape)

        delta = cic_paint(jnp.zeros(mesh_shape), pos)

        delta_k = jnp.fft.rfftn(delta)

        # Computes gravitational potential
        pot_k = delta_k * invlaplace_kernel(kvec) * longrange_kernel(kvec, r_split=0)

        # Apply a correction filter
        kk = jnp.sqrt(sum((ki/jnp.pi)**2 for ki in kvec))
        pot_k = pot_k *(1. + model.apply(params, kk, jnp.atleast_1d(a)))

        # Computes gravitational forces
        forces = jnp.stack([cic_read(jnp.fft.irfftn(- gradient_kernel(kvec, i)*pot_k), pos) 
                          for i in range(3)],axis=-1)

        forces = forces * 1.5 * cosmo.Omega_m

        # Computes the update of position (drift)
        dpos = 1. / (a**3 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * vel

        # Computes the update of velocity (kick)
        dvel = 1. / (a**2 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * forces

        return dpos, dvel
    return neural_nbody_ode
