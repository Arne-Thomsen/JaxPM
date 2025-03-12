import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import norm

from jaxpm.painting import cic_paint

__all__ = ["power_spectrum"]


def _initialize_pk(shape, boxsize, kmin, dk):
    """
    Helper function to initialize various (fixed) values for powerspectra... not differentiable!
    """
    I = np.eye(len(shape), dtype="int") * -2 + 1

    W = np.empty(shape, dtype="f4")
    W[...] = 2.0
    W[..., 0] = 1.0
    W[..., -1] = 1.0

    kmax = np.pi * np.min(np.array(shape)) / np.max(np.array(boxsize)) + dk / 2
    kedges = np.arange(kmin, kmax, dk)

    k = [
        np.fft.fftfreq(N, 1.0 / (N * 2 * np.pi / L))[:pkshape].reshape(kshape)
        for N, L, kshape, pkshape in zip(shape, boxsize, I, shape)
    ]
    kmag = sum(ki**2 for ki in k) ** 0.5

    xsum = np.zeros(len(kedges) + 1)
    Nsum = np.zeros(len(kedges) + 1)

    dig = np.digitize(kmag.flat, kedges)

    xsum.flat += np.bincount(dig, weights=(W * kmag).flat, minlength=xsum.size)
    Nsum.flat += np.bincount(dig, weights=W.flat, minlength=xsum.size)
    return dig, Nsum, xsum, W, k, kedges


def power_spectrum(field, kmin=5, dk=0.5, boxsize=False):
    """
    Calculate the powerspectra given real space field

    Args:

        field: real valued field
        kmin: minimum k-value for binned powerspectra
        dk: differential in each kbin
        boxsize: length of each boxlength (can be strangly shaped?)

    Returns:

        kbins: the central value of the bins for plotting
        power: real valued array of power in each bin

    """
    shape = field.shape
    nx, ny, nz = shape

    # initialze values related to powerspectra (mode bins and weights)
    dig, Nsum, xsum, W, k, kedges = _initialize_pk(shape, boxsize, kmin, dk)

    # fast fourier transform
    fft_image = jnp.fft.fftn(field)

    # absolute value of fast fourier transform
    pk = jnp.real(fft_image * jnp.conj(fft_image))

    # calculating powerspectra
    real = jnp.real(pk).reshape([-1])
    imag = jnp.imag(pk).reshape([-1])

    Psum = jnp.bincount(dig, weights=(W.flatten() * imag), length=xsum.size) * 1j
    Psum += jnp.bincount(dig, weights=(W.flatten() * real), length=xsum.size)

    P = ((Psum / Nsum)[1:-1] * boxsize.prod()).astype("float32")

    # normalization for powerspectra
    norm = np.prod(np.array(shape[:])).astype("float32") ** 2

    # find central values of each bin
    kbins = kedges[:-1] + (kedges[1:] - kedges[:-1]) / 2

    return kbins, P / norm


def cross_correlation_coefficients(field_a, field_b, kmin=5, dk=0.5, boxsize=False):
    """
    Calculate the cross correlation coefficients given two real space field

    Args:

        field_a: real valued field
        field_b: real valued field
        kmin: minimum k-value for binned powerspectra
        dk: differential in each kbin
        boxsize: length of each boxlength (can be strangly shaped?)

    Returns:

        kbins: the central value of the bins for plotting
        P / norm: normalized cross correlation coefficient between two field a and b

    """
    shape = field_a.shape
    nx, ny, nz = shape

    # initialze values related to powerspectra (mode bins and weights)
    dig, Nsum, xsum, W, k, kedges = _initialize_pk(shape, boxsize, kmin, dk)

    # fast fourier transform
    fft_image_a = jnp.fft.fftn(field_a)
    fft_image_b = jnp.fft.fftn(field_b)

    # absolute value of fast fourier transform
    pk = fft_image_a * jnp.conj(fft_image_b)

    # calculating powerspectra
    real = jnp.real(pk).reshape([-1])
    imag = jnp.imag(pk).reshape([-1])

    Psum = jnp.bincount(dig, weights=(W.flatten() * imag), length=xsum.size) * 1j
    Psum += jnp.bincount(dig, weights=(W.flatten() * real), length=xsum.size)

    P = ((Psum / Nsum)[1:-1] * boxsize.prod()).astype("float32")

    # normalization for powerspectra
    norm = np.prod(np.array(shape[:])).astype("float32") ** 2

    # find central values of each bin
    kbins = kedges[:-1] + (kedges[1:] - kedges[:-1]) / 2

    return kbins, P / norm


def gaussian_smoothing(im, sigma):
    """
    im: 2d image
    sigma: smoothing scale in px
    """
    # Compute k vector
    kvec = jnp.stack(jnp.meshgrid(jnp.fft.fftfreq(im.shape[0]), jnp.fft.fftfreq(im.shape[1])), axis=-1)
    k = jnp.linalg.norm(kvec, axis=-1)
    # We compute the value of the filter at frequency k
    filter = norm.pdf(k, 0, 1.0 / (2.0 * np.pi * sigma))
    filter /= filter[0, 0]

    return jnp.fft.ifft2(jnp.fft.fft2(im) * filter).real


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
