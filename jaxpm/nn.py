import haiku as hk
import jax
import jax.numpy as jnp
from flax import nnx

from tqdm import tqdm


def _deBoorVectorized(x, knot_positions, control_points, degree):
    """
    Evaluates the B-spline at a given position using the de Boor algorithm.

    Args:
    -----
    x : float
        The position at which to evaluate the B-spline.
    knot_positions : jnp.ndarray
        Array of knot positions, needs to be padded appropriately.
    control_points : jnp.ndarray
        Array of control points.
    degree : int
        Degree of the B-spline.

    Returns:
    --------
    float
        The evaluated value of the B-spline at position x.
    """
    k = jnp.digitize(x, knot_positions) - 1

    d = [control_points[j + k - degree] for j in range(0, degree + 1)]
    for r in range(1, degree + 1):
        for j in range(degree, r - 1, -1):
            alpha = (x - knot_positions[j + k - degree]) / (
                knot_positions[j + 1 + k - r] - knot_positions[j + k - degree]
            )
            d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j]
    return d[degree]


class NeuralSplineFourierFilter(hk.Module):
    """A rotationally invariant filter parameterized by
    a b-spline with parameters specified by a small NN."""

    def __init__(self, n_knots=8, latent_size=16, name=None):
        """
        n_knots: number of control points for the spline
        """
        super().__init__(name=name)
        self.n_knots = n_knots
        self.latent_size = latent_size

    def __call__(self, x, a):
        """
        x: array, scale, normalized to fftfreq default
        a: scalar, scale factor
        """

        net = jnp.sin(hk.Linear(self.latent_size)(jnp.atleast_1d(a)))
        net = jnp.sin(hk.Linear(self.latent_size)(net))

        w = hk.Linear(self.n_knots + 1)(net)
        k = hk.Linear(self.n_knots - 1)(net)

        # make sure the knots sum to 1 and are in the interval 0,1
        k = jnp.concatenate([jnp.zeros((1,)), jnp.cumsum(jax.nn.softmax(k))])

        w = jnp.concatenate([jnp.zeros((1,)), w])

        # Augment with repeating points
        ak = jnp.concatenate([jnp.zeros((3,)), k, jnp.ones((3,))])

        return _deBoorVectorized(jnp.clip(x / jnp.sqrt(3), 0, 1 - 1e-4), ak, w, 3)


class NeuralSplineFourierFilterNNX(nnx.Module):
    """A rotationally invariant filter parameterized by
    a b-spline with parameters specified by a small NN."""

    def __init__(self, n_knots: int, d_latent: int, rngs: nnx.Rngs):
        """Initialize the filter with number of knots and latent dimension."""
        super().__init__()
        self.n_knots = n_knots
        self.d_latent = d_latent

        self.linear_a1 = nnx.Linear(1, self.d_latent, rngs=rngs)
        self.linear_a2 = nnx.Linear(self.d_latent, self.d_latent, rngs=rngs)
        self.linear_w = nnx.Linear(self.d_latent, self.n_knots + 1, rngs=rngs)
        self.linear_k = nnx.Linear(self.d_latent, self.n_knots - 1, rngs=rngs)

    def __call__(self, x, a):
        """
        x: array, scale, normalized to fftfreq default
        a: scalar, scale factor
        """
        # Embed the scale factor a
        net = jnp.sin(self.linear_a1(jnp.atleast_1d(a)))
        net = jnp.sin(self.linear_a2(net))

        # Generate spline parameters
        w = self.linear_w(net)
        k = self.linear_k(net)

        # Ensure knots sum to 1 and are in interval [0,1]
        k = jnp.concatenate([jnp.zeros((1,)), jnp.cumsum(jax.nn.softmax(k))])
        w = jnp.concatenate([jnp.zeros((1,)), w])

        # Augment with repeating points for B-spline
        ak = jnp.concatenate([jnp.zeros((3,)), k, jnp.ones((3,))])

        return _deBoorVectorized(jnp.clip(x / jnp.sqrt(3), 0, 1 - 1e-4), ak, w, 3)


class MLP(nnx.Module):
    def __init__(self, d_in: int, d_out: int, d_hidden: int, n_hidden: int, rngs: nnx.Rngs, activation=jax.nn.relu):
        self.linear_in = nnx.Linear(d_in, d_hidden, rngs=rngs)
        self.linear_hid = [nnx.Linear(d_hidden, d_hidden, rngs=rngs) for _ in range(n_hidden)]
        self.linear_out = nnx.Linear(d_hidden, d_out, rngs=rngs)
        self.activation = activation

    def __call__(self, x):
        x = self.activation(self.linear_in(x))
        for layer in self.linear_hid:
            x = self.activation(layer(x))
        x = self.linear_out(x)
        return x


def batched_eval(model, in_array, batch_size):
    assert in_array.ndim == 2

    preds = []
    for i in tqdm(range(in_array.shape[0] // batch_size)):
        preds.append(model(in_array[i * batch_size : (i + 1) * batch_size]))
    preds.append(model(in_array[(i + 1) * batch_size :]))

    return jnp.concatenate(preds, axis=0)
