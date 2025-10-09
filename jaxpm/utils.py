import numpy as np

import jax.numpy as jnp
from jax.scipy.stats import norm
import orbax.checkpoint as ocp
from flax import nnx


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


def refine_time_steps(t, n):
    """
    Given non-uniform time steps t, add n evenly spaced points between each consecutive pair.

    Args:
        t: 1D jax.numpy array of shape (m,)
        n: number of points to insert between each pair

    Returns:
        jnp.ndarray of shape (m-1)*(n+1) + 1
    """
    # For each interval, make linspace with n+2 points, then drop the last one
    refined_segments = [jnp.linspace(t[i], t[i + 1], n + 2)[:-1] for i in range(len(t) - 1)]

    # Concatenate all intervals and append the final endpoint
    return jnp.concatenate(refined_segments + [t[-1:]])


class BaseModel(nnx.Module):
    def __init__(self):
        self.checkpointer = ocp.StandardCheckpointer()

    def __call__(self, x, training: bool = False):
        raise NotImplementedError("BaseModel is an abstract class and cannot be called directly.")

    def load(self, checkpoint_file):
        abstract_model = nnx.eval_shape(lambda: self)
        graphdef, abstract_params = nnx.split(abstract_model)

        params = self.checkpointer.restore(checkpoint_file, abstract_params)
        # update in place
        self.__dict__.update(nnx.merge(graphdef, params).__dict__)
        print(f"Checkpoint loaded from {checkpoint_file}")

    def save(self, checkpoint_file):
        _, params = nnx.split(self)
        self.checkpointer.save(checkpoint_file, params, force=True)
        print(f"Checkpoint saved to {checkpoint_file}")
