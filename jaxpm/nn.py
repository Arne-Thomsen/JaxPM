import jax
import jax.numpy as jnp

import haiku as hk
from flax import nnx
from jraph import GraphConvolution, GAT

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


class Flatten(nnx.Module):
    def __call__(self, x):
        return x.reshape((x.shape[0], -1))


class ResNetBlock3D(nnx.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: tuple = (3, 3, 3),
        strides: int = 1,
        activation=jax.nn.relu,
        rngs: nnx.Rngs = None,
    ):
        self.filters = channels
        self.strides = strides
        self.activation = activation

        self.conv1 = nnx.Conv(channels, channels, kernel_size, strides, padding="SAME", rngs=rngs)
        self.conv2 = nnx.Conv(channels, channels, kernel_size, 1, padding="SAME", rngs=rngs)
        self.convres = nnx.Conv(channels, channels, (1, 1, 1), strides, padding="SAME", rngs=rngs)

        self.norm1 = nnx.BatchNorm(channels, rngs=rngs)
        self.norm2 = nnx.BatchNorm(channels, rngs=rngs)
        self.normres = nnx.BatchNorm(channels, rngs=rngs)

    def __call__(self, x, training: bool = False):
        residual = x
        y = self.conv1(x)
        # y = self.norm1(y, use_running_average=not training)
        y = self.activation(y)
        y = self.conv2(y)
        # y = self.norm2(y, use_running_average=not training)

        if residual.shape != y.shape:
            residual = self.convres(residual)
            residual = self.normres(residual, use_running_average=not training)
            # residual = self.norm(name='norm_proj')(residual)

        return self.activation(residual + y)


class ResNet3D(nnx.Module):
    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        d_out: int,
        num_blocks: int,
        rngs: nnx.Rngs,
        kernel_size: tuple = (3, 3, 3),
        strides: int = 1,
    ):
        self.conv_in = nnx.Conv(d_in, d_hidden, kernel_size, 1, padding="SAME", rngs=rngs)
        self.blocks = [ResNetBlock3D(d_hidden, kernel_size, strides, rngs=rngs) for _ in range(num_blocks)]
        self.conv_out = nnx.Conv(d_hidden, d_out, kernel_size, 1, padding="SAME", rngs=rngs)

        # self.norm = nnx.BatchNorm()

        # self.flatten = Flatten()
        # # self.linear_hidden = nnx.Linear(d_hidden, d_hidden, rngs=rngs)
        # # self.linear_out = nnx.Linear(d_out, d_out, rngs=rngs)
        # # TODO
        # self.linear_hidden = nnx.Linear(8192, 64, rngs=rngs)
        # self.linear_out = nnx.Linear(64, d_out, rngs=rngs)

    def __call__(self, x, training: bool = False):
        x = self.conv_in(x)
        # x = self.norm(x, use_running_average=not training)
        x = jax.nn.relu(x)
        for block in self.blocks:
            x = block(x, training=training)

        # x = self.flatten(x)
        # x = self.linear_hidden(x)
        # x = self.linear_out(x)

        x = self.conv_out(x)
        x = jnp.squeeze(x)

        return x


def batched_eval(model, in_array, batch_size):
    assert in_array.ndim == 2

    preds = []
    for i in tqdm(range(in_array.shape[0] // batch_size)):
        preds.append(model(in_array[i * batch_size : (i + 1) * batch_size]))
    preds.append(model(in_array[(i + 1) * batch_size :]))

    return jnp.concatenate(preds, axis=0)


class GNN(nnx.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        d_hidden: int,
        n_hidden: int,
        rngs: nnx.Rngs,
        activation=jax.nn.relu,
        normalize=True,
    ):
        super().__init__()
        self.linear_in = nnx.Linear(d_in, d_hidden, rngs=rngs)
        self.linear_hid = [nnx.Linear(d_hidden, d_hidden, rngs=rngs) for _ in range(n_hidden)]
        self.linear_out = nnx.Linear(d_hidden, d_out, rngs=rngs)
        self.activation = activation
        self.normalize = normalize

        self.graph_convolution = lambda graph, update_node_fn: GraphConvolution(
            update_node_fn=update_node_fn,
            symmetric_normalization=self.normalize,
        )(graph)

    def __call__(self, graph):
        graph = self.graph_convolution(graph, update_node_fn=lambda n: self.activation(self.linear_in(n)))
        for linear in self.linear_hid:
            graph = self.graph_convolution(graph, update_node_fn=lambda n: self.activation(linear(n)))
        graph = self.graph_convolution(graph, update_node_fn=lambda n: self.linear_out(n))

        return graph


class GATGNN(nnx.Module):
    def __init__(
        self,
        d_node: int,
        d_edge: int,
        d_query: int,
        d_out: int,
        n_hidden: int,
        rngs: nnx.Rngs,
        activation=jax.nn.relu,
    ):
        super().__init__()

        self.query_in = nnx.Linear(d_node, d_query, rngs=rngs)
        self.logit_in = nnx.Linear(2 * d_query + d_edge, d_query, rngs=rngs)

        self.query_hid = [nnx.Linear(d_query, d_query, rngs=rngs) for _ in range(n_hidden)]
        self.logit_hid = [nnx.Linear(2 * d_query + d_edge, d_query, rngs=rngs) for _ in range(n_hidden)]

        self.query_out = nnx.Linear(d_query, d_out, rngs=rngs)
        self.logit_out = nnx.Linear(2 * d_out + d_edge, d_out, rngs=rngs)

        self.activation = activation

        self.gat = lambda graph, query_layer, logit_layer: GAT(
            attention_query_fn=self.get_query_fn(query_layer),
            attention_logit_fn=self.get_logit_fn(logit_layer),
            node_update_fn=None,
        )(graph)

    def get_logit_fn(self, layer, apply_activation=False):
        def logit_fn(sender_features, receiver_features, edge_features):
            concatenated_features = jnp.concatenate([sender_features, receiver_features, edge_features], axis=-1)
            logits = layer(concatenated_features)
            if apply_activation:
                logits = self.activation(logits)
            return logits

        return logit_fn

    def get_query_fn(self, layer, apply_activation=False):
        def query_fn(node_features):
            query = layer(node_features)
            if apply_activation:
                query = self.activation(query)
            return query

        return query_fn

    def __call__(self, graph):
        graph = self.gat(graph, self.query_in, self.logit_in)
        for query, logit in zip(self.query_hid, self.logit_hid):
            graph = self.gat(graph, query, logit)
        graph = self.gat(graph, self.query_out, self.logit_out)

        return graph
