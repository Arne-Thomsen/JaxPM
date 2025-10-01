import jax
import jax.numpy as jnp

from jaxpm.painting import cic_paint, cic_read

import haiku as hk
from flax import nnx
import flax.linen as nn
from jraph import GraphConvolution, GAT

from tqdm import tqdm


def batched_eval(model, in_array, batch_size):
    assert in_array.ndim == 2

    preds = []
    for i in tqdm(range(in_array.shape[0] // batch_size)):
        preds.append(model(in_array[i * batch_size : (i + 1) * batch_size]))
    preds.append(model(in_array[(i + 1) * batch_size :]))

    return jnp.concatenate(preds, axis=0)


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

    def __call__(self, x, a, eps=1e-4, training=False):
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

        return _deBoorVectorized(jnp.clip(x / jnp.sqrt(3), 0, 1 - eps), ak, w, 3)


class MLP(nnx.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        d_hidden: int,
        n_hidden: int,
        rngs: nnx.Rngs,
        dropout_rate: float = 0.0,
        activation=jax.nn.relu,
        norm_type: str = "layer",
        k_filter: str = None,
    ):
        self.linear_in = nnx.Linear(d_in, d_hidden, rngs=rngs)
        self.linear_hid = [nnx.Linear(d_hidden, d_hidden, rngs=rngs) for _ in range(n_hidden)]
        self.linear_out = nnx.Linear(d_hidden, d_out, rngs=rngs)
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.norm_type = norm_type

        if k_filter == "gaussian":
            self.k_smooth = nnx.Param(jnp.array(1.0))
        elif k_filter == "spline":
            self.k_model = NeuralSplineFourierFilterNNX(n_knots=8, d_latent=16, rngs=rngs)
        elif k_filter is not None:
            raise ValueError(f"Unsupported k_filter: {k_filter}")

        if isinstance(self.activation, str):
            if self.activation == "relu":
                self.activation = jax.nn.relu
            elif self.activation == "swish":
                self.activation = jax.nn.swish
            elif self.activation == "sigmoid":
                self.activation = jax.nn.sigmoid
            else:
                raise ValueError(f"Unsupported activation function: {self.activation}")

        if self.dropout_rate > 0:
            self.dropout = [nnx.Dropout(dropout_rate, rngs=rngs) for _ in range(n_hidden)]

        if self.norm_type == "layer":
            self.norm_in = nnx.LayerNorm(d_hidden, rngs=rngs)
            self.norm_hid = [nnx.LayerNorm(d_hidden, rngs=rngs) for _ in range(n_hidden)]
        elif self.norm_type == "batch":
            self.norm_in = nnx.BatchNorm(d_hidden, rngs=rngs)
            self.norm_hid = [nnx.BatchNorm(d_hidden, rngs=rngs) for _ in range(n_hidden)]

        self.d_out = d_out

    def __call__(self, x, training: bool = False):
        x = self.linear_in(x)

        if self.norm_type == "layer":
            x = self.norm_in(x)
        elif self.norm_type == "batch":
            x = self.norm_in(x, use_running_average=not training)

        x = self.activation(x)

        for i, linear in enumerate(self.linear_hid):
            x = linear(x)

            if self.norm_type == "layer":
                x = self.norm_hid[i](x)
            elif self.norm_type == "batch":
                x = self.norm_hid[i](x, use_running_average=not training)

            x = self.activation(x)
            if training and self.dropout_rate > 0:
                x = self.dropout[i](x, deterministic=not training)

        x = self.linear_out(x)
        return x


class ConditionedCNN(nnx.Module):
    def __init__(
        self,
        d_in=4,
        d_hidden=64,
        d_out=1,
        n_hidden=4,
        kernel_size=(3, 3, 3),
        activation=jax.nn.swish,
        use_residual=True,
        norm_type="layer",
        group_norm_groups: int | None = None,
        rngs=nnx.Rngs(0),
    ):
        self.activation = activation
        self.use_residual = use_residual
        self.norm_type = norm_type
        self.d_out = d_out

        # CNN
        self.conv_in = nnx.Conv(d_in, d_hidden, kernel_size, padding="CIRCULAR", rngs=rngs)
        self.conv_hidden = [
            nnx.Conv(d_hidden, d_hidden, kernel_size, padding="CIRCULAR", rngs=rngs) for _ in range(n_hidden)
        ]
        self.conv_out = nnx.Conv(d_hidden, d_out, kernel_size, padding="CIRCULAR", rngs=rngs)

        if self.norm_type == "layer":
            self.norm_in = nnx.LayerNorm(d_hidden, rngs=rngs)
            self.norm_hidden = [nnx.LayerNorm(d_hidden, rngs=rngs) for _ in range(n_hidden)]
        elif self.norm_type == "batch":
            self.norm_in = nnx.BatchNorm(d_hidden, rngs=rngs)
            self.norm_hidden = [nnx.BatchNorm(d_hidden, rngs=rngs) for _ in range(n_hidden)]
        elif self.norm_type == "group":
            # Choose a reasonable default number of groups for 3D CNNs
            def _pick_groups(c: int, desired: int = 32) -> int:
                g = min(desired, c)
                # Ensure divisibility
                while c % g != 0 and g > 1:
                    g -= 1
                return g

            gn_groups = group_norm_groups or _pick_groups(d_hidden)
            # Normalize over spatial dims only (channels-last), excluding batch and channel
            # Works for both (H, W, D, C) and (B, H, W, D, C)
            reduction_axes = (-4, -3, -2)
            self.norm_in = nnx.GroupNorm(d_hidden, num_groups=gn_groups, reduction_axes=reduction_axes, rngs=rngs)
            self.norm_hidden = [
                nnx.GroupNorm(d_hidden, num_groups=gn_groups, reduction_axes=reduction_axes, rngs=rngs)
                for _ in range(n_hidden)
            ]

        # scale conditioning https://arxiv.org/abs/1709.07871
        self.scale_embed = nnx.Linear(1, d_hidden, rngs=rngs)
        self.film_gamma = [nnx.Linear(d_hidden, d_hidden, rngs=rngs) for _ in range(n_hidden)]
        self.film_beta = [nnx.Linear(d_hidden, d_hidden, rngs=rngs) for _ in range(n_hidden)]

    def __call__(self, x, scale, training=False):
        """Forward pass through the conditioned CNN.

        Args:
            x: Input tensor of shape (mesh, mesh, mesh, features) or (batch, mesh, mesh, mesh, features)
            scale: Scale factor(s), shape () for single scale or (batch,) for batched scales
            training: Whether in training mode

        Returns:
            Output tensor of same spatial shape as input
        """
        # Determine if input is batched
        is_batched = x.ndim == 5  # (batch, H, W, D, features) vs (H, W, D, features)

        x = self.conv_in(x)

        if self.norm_type == "layer":
            x = self.norm_in(x)
        elif self.norm_type == "batch":
            x = self.norm_in(x, use_running_average=not training)
        elif self.norm_type == "group":
            x = self.norm_in(x)

        x = self.activation(x)

        # Handle scale embedding for both batched and non-batched inputs
        if is_batched:
            # Ensure scale has batch dimension
            if scale.ndim == 0:  # Single scale for all batch items
                scale = jnp.broadcast_to(scale, (x.shape[0],))
            elif scale.ndim == 1 and scale.shape[0] != x.shape[0]:
                raise ValueError(f"Scale batch size {scale.shape[0]} doesn't match input batch size {x.shape[0]}")

            scale_embedding = self.activation(self.scale_embed(scale.reshape(-1, 1)))
        else:
            # Non-batched case
            scale_embedding = self.activation(self.scale_embed(scale.reshape(1, 1)))

        # Hidden layers with scale conditioning
        for i, conv in enumerate(self.conv_hidden):
            if self.use_residual:
                residual = x

            x = conv(x)

            if self.norm_type == "layer":
                x = self.norm_hidden[i](x)
            elif self.norm_type == "batch":
                x = self.norm_hidden[i](x, use_running_average=not training)
            elif self.norm_type == "group":
                x = self.norm_hidden[i](x)

            # Apply FiLM conditioning (scale and shift based on scale factor)
            gamma = self.film_gamma[i](scale_embedding)
            beta = self.film_beta[i](scale_embedding)

            # Reshape for broadcasting based on input dimensionality
            if is_batched:
                # Shape: (batch, 1, 1, 1, d_hidden)
                gamma = gamma.reshape(gamma.shape[0], 1, 1, 1, -1)
                beta = beta.reshape(beta.shape[0], 1, 1, 1, -1)
            else:
                # Shape: (1, 1, 1, d_hidden)
                gamma = gamma.reshape(1, 1, 1, -1)
                beta = beta.reshape(1, 1, 1, -1)

            # Apply conditioning
            x = x * gamma + beta

            if self.use_residual:
                x = x + residual

            x = self.activation(x)

        x = self.conv_out(x)
        return x


# deprecated ##########################################################################################################


class ResidualMLP(nnx.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        d_hidden: int,
        n_blocks: int,
        rngs: nnx.Rngs,
        dropout_rate: float = 0.0,
        activation=jax.nn.relu,
        norm_type: str = "batch",
    ):
        """
        Initialize a Residual MLP with skip connections.

        Args:
            d_in: Input dimension
            d_out: Output dimension
            d_hidden: Hidden dimension used throughout the network
            n_blocks: Number of residual blocks
            rngs: Random number generators
            dropout_rate: Dropout rate (if > 0)
            activation: Activation function
            norm_type: Normalization type ("batch", "layer" or None)
        """
        # Input projection
        self.input_proj = nnx.Linear(d_in, d_hidden, rngs=rngs)

        # Residual blocks
        self.blocks = []
        for _ in range(n_blocks):
            block = {
                "linear1": nnx.Linear(d_hidden, d_hidden, rngs=rngs),
                "linear2": nnx.Linear(d_hidden, d_hidden, rngs=rngs),
            }

            if norm_type == "layer":
                block["norm1"] = nnx.LayerNorm(d_hidden, rngs=rngs)
                block["norm2"] = nnx.LayerNorm(d_hidden, rngs=rngs)
            elif norm_type == "batch":
                block["norm1"] = nnx.BatchNorm(d_hidden, rngs=rngs)
                block["norm2"] = nnx.BatchNorm(d_hidden, rngs=rngs)

            if dropout_rate > 0:
                block["dropout"] = nnx.Dropout(dropout_rate, rngs=rngs)

            self.blocks.append(block)

        # Output projection
        self.output_proj = nnx.Linear(d_hidden, d_out, rngs=rngs)

        # Store parameters
        self.activation = activation
        self.norm_type = norm_type
        self.dropout_rate = dropout_rate
        self.d_out = d_out

    def __call__(self, x, training: bool = False):
        # Input projection
        x = self.input_proj(x)

        # Process through residual blocks
        for block in self.blocks:
            # Store the input for the skip connection
            residual = x

            # First layer
            x = block["linear1"](x)
            if self.norm_type == "layer":
                x = block["norm1"](x)
            elif self.norm_type == "batch":
                x = block["norm1"](x, use_running_average=not training)
            x = self.activation(x)

            # Second layer
            x = block["linear2"](x)
            if self.norm_type == "layer":
                x = block["norm2"](x)
            elif self.norm_type == "batch":
                x = block["norm2"](x, use_running_average=not training)

            # Apply dropout if needed
            if training and self.dropout_rate > 0:
                x = block["dropout"](x, deterministic=not training)

            # Add the residual connection
            x = x + residual

            # Apply activation after the residual connection
            x = self.activation(x)

        # Output projection
        x = self.output_proj(x)

        return x


class HybridNet(nnx.Module):
    def __init__(
        self,
        mlp,
        cnn,
        d_out,
        rngs,
        batch_axis=False,
    ):
        self.mlp = mlp
        self.cnn = cnn
        self.linear_out = nnx.Linear(self.mlp.d_out + self.cnn.d_out, d_out, rngs=rngs)

        self.vcic_read = jax.vmap(
            cic_read,
            # feature dimension
            in_axes=(-1, None),
            out_axes=-1,
        )
        if batch_axis:
            self.vcic_read = jax.vmap(self.vcic_read, in_axes=(0, 0))

    def __call__(self, pos, particle, field):
        particle = self.mlp(particle)

        field = self.cnn(field)
        field = self.vcic_read(field, pos)

        x = jnp.concatenate([particle, field], axis=-1)
        x = self.linear_out(x)

        return x


class ParticleToParticleNet(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features_mlp: int,
        hidden_features_cnn: int,
        num_layers_mlp: int,
        num_layers_cnn: int,
        rngs: nnx.Rngs,
        dropout_rate: float = 0.0,
        activation=jax.nn.relu,
        norm_type: str = "batch",
        batch_axis: bool = False,
    ):
        self.mlp = MLP(
            in_features,
            hidden_features_mlp,
            out_features,
            num_layers_mlp,
            rngs=rngs,
            dropout_rate=dropout_rate,
            activation=activation,
            norm_type=norm_type,
        )

        self.cnn = ConditionedCNN(
            d_in=in_features,
            d_hidden=hidden_features_cnn,
            d_out=out_features,
            n_hidden=num_layers_cnn,
            rngs=rngs,
            activation=activation,
            norm_type=norm_type,
        )

        self.linear_out = nnx.Linear(self.mlp.d_out + self.cnn.d_out, out_features, rngs=rngs)

        # feature dimension
        self.vcic_paint = jax.vmap(cic_paint, in_axes=(None, None, -1), out_axes=-1)
        if batch_axis:
            self.vcic_paint = jax.vmap(self.vcic_paint, in_axes=(0, 0, 0))

        # feature dimension
        self.vcic_read = jax.vmap(cic_read, in_axes=(-1, None), out_axes=-1)
        if batch_axis:
            self.vcic_read = jax.vmap(self.vcic_read, in_axes=(0, 0))

    def __call__(self, x, training: bool = False):
        pass


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

        self.conv1 = nnx.Conv(channels, channels, kernel_size, strides, padding="CIRCULAR", rngs=rngs)
        self.conv2 = nnx.Conv(channels, channels, kernel_size, 1, padding="CIRCULAR", rngs=rngs)
        self.convres = nnx.Conv(channels, channels, (1, 1, 1), strides, padding="CIRCULAR", rngs=rngs)

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


class Flatten(nnx.Module):
    def __call__(self, x):
        return x.reshape((x.shape[0], -1))


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
        self.conv_in = nnx.Conv(d_in, d_hidden, kernel_size, 1, padding="CIRCULAR", rngs=rngs)
        self.blocks = [ResNetBlock3D(d_hidden, kernel_size, strides, rngs=rngs) for _ in range(num_blocks)]
        self.conv_out = nnx.Conv(d_hidden, d_out, kernel_size, 1, padding="CIRCULAR", rngs=rngs)

        # self.norm = nnx.BatchNorm()

        self.flatten = Flatten()
        # # self.linear_hidden = nnx.Linear(d_hidden, d_hidden, rngs=rngs)
        # # self.linear_out = nnx.Linear(d_out, d_out, rngs=rngs)
        # # TODO
        # self.linear_hidden = nnx.Linear(8192, 64, rngs=rngs)
        # self.linear_hidden = nnx.Linear(262144, 64, rngs=rngs)
        self.linear_hidden = nnx.Linear(4096, 64, rngs=rngs)
        self.linear_out = nnx.Linear(64, d_out, rngs=rngs)

    def __call__(self, x, training: bool = False):
        x = self.conv_in(x)
        # x = self.norm(x, use_running_average=not training)
        x = jax.nn.relu(x)
        for block in self.blocks:
            x = block(x, training=training)

        x = self.conv_out(x)
        x = jnp.squeeze(x)

        x = self.flatten(x)
        x = self.linear_hidden(x)
        x = self.linear_out(x)

        return x


class ConvGNN(nnx.Module):
    def __init__(
        self,
        d_node: int,
        d_out: int,
        d_hidden: int,
        n_hidden: int,
        rngs: nnx.Rngs,
        activation=jax.nn.relu,
        normalize=True,
    ):
        super().__init__()
        self.linear_in = nnx.Linear(d_node, d_hidden, rngs=rngs)
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


class AttentionGNN(nnx.Module):
    def __init__(
        self,
        d_node: int,
        d_edge: int,
        d_query: int,
        d_out: int,
        n_hidden: int,
        rngs: nnx.Rngs,
        activation=jax.nn.relu,
        query_activation=False,
        logit_activation=False,
        final_projection=False,
    ):
        super().__init__()

        self.query_in = nnx.Linear(d_node, d_query, rngs=rngs)
        self.logit_in = nnx.Linear(2 * d_query + d_edge, d_query, rngs=rngs)

        self.query_hid = [nnx.Linear(d_query, d_query, rngs=rngs) for _ in range(n_hidden)]
        self.logit_hid = [nnx.Linear(2 * d_query + d_edge, d_query, rngs=rngs) for _ in range(n_hidden)]

        self.final_projection = final_projection
        if self.final_projection:
            self.query_out = nnx.Linear(d_query, d_query, rngs=rngs)
            self.logit_out = nnx.Linear(2 * d_query + d_edge, d_query, rngs=rngs)
            self.linear_out = nnx.Linear(d_query, d_out, rngs=rngs)
        else:
            self.query_out = nnx.Linear(d_query, d_out, rngs=rngs)
            self.logit_out = nnx.Linear(2 * d_out + d_edge, d_out, rngs=rngs)

        self.activation = activation

        self.gat = lambda graph, query_layer, logit_layer: GAT(
            attention_query_fn=self.get_query_fn(query_layer, query_activation),
            attention_logit_fn=self.get_logit_fn(logit_layer, logit_activation),
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

    # def get_update_fn(self, layer)

    def __call__(self, graph, training=False):
        graph = self.gat(graph, self.query_in, self.logit_in)
        for query, logit in zip(self.query_hid, self.logit_hid):
            graph = self.gat(graph, query, logit)
        graph = self.gat(graph, self.query_out, self.logit_out)

        if self.final_projection:
            graph = graph._replace(nodes=self.linear_out(graph.nodes))

        return graph.nodes
