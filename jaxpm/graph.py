import numpy as np
import scipy.spatial

import jax
import jax.numpy as jnp
import jraph

from functools import partial


def scipy_get_knn(points, k, distance_upper_bound=np.inf, boxsize=None, workers=-1, leafsize=10):
    kd_tree = scipy.spatial.cKDTree(data=points, boxsize=boxsize, leafsize=leafsize)
    distances, idx = kd_tree.query(x=points, k=int(k), workers=workers, distance_upper_bound=distance_upper_bound)
    return distances.astype(points.dtype), idx.astype(np.int32)


@partial(jax.jit, static_argnames=["k", "distance_upper_bound", "boxsize", "workers", "leafsize"])
def jax_get_knn(points, k, distance_upper_bound=np.inf, boxsize=None, workers=-1, leafsize=10):
    shape = (jnp.shape(points)[0], k)
    distance_type = jax.ShapeDtypeStruct(shape, points.dtype)
    idx_type = jax.ShapeDtypeStruct(shape, jnp.int32)
    return jax.pure_callback(
        scipy_get_knn,
        (distance_type, idx_type),
        points,
        k,
        distance_upper_bound,
        boxsize,
        workers,
        leafsize,
    )


# def get_edges(poss, scales, k=4):
#     if poss.ndim == 3:
#         assert poss.shape[0] == scales.shape[0]

#     n_node = poss.shape[1]

#     # TODO vectorize this properly
#     features, senders, receivers = [], [], []
#     for i in range(poss.shape[0]):
#         k_dist, k_idx = jax_get_knn(poss[i], k)
#         feature = k_dist.reshape(-1, 1)
#         sender = k_idx.reshape(-1)
#         receiver = jnp.repeat(jnp.arange(n_node, dtype=jnp.int32), k)

#         features.append(feature)
#         senders.append(sender)
#         receivers.append(receiver)

#     edges = {}
#     edges["features"] = jnp.stack(features, axis=0)
#     edges["senders"] = jnp.stack(senders, axis=0)
#     edges["receivers"] = jnp.stack(receivers, axis=0)
#     edges["scales"] = scales

#     return edges


def get_edges(poss, scales, k=4, boxsize=None):
    def get_edges_single(pos):
        k_dist, k_idx = jax_get_knn(pos, k, boxsize=boxsize)
        feature = k_dist.reshape(-1, 1)
        sender = k_idx.reshape(-1)
        receiver = jnp.repeat(jnp.arange(pos.shape[0], dtype=jnp.int32), k)

        return feature, sender, receiver

    edges = {}
    if poss.ndim == 3:
        assert poss.shape[0] == scales.shape[0]

        # TODO vectorize this properly
        features, senders, receivers = [], [], []
        for i in range(poss.shape[0]):
            feature, sender, receiver = get_edges_single(poss[i])

            features.append(feature)
            senders.append(sender)
            receivers.append(receiver)

        features = jnp.stack(features, axis=0)
        senders = jnp.stack(senders, axis=0)
        receivers = jnp.stack(receivers, axis=0)

    elif poss.ndim == 2:
        features, senders, receivers = get_edges_single(poss)

    edges["features"] = features
    edges["senders"] = senders
    edges["receivers"] = receivers
    edges["scales"] = scales

    return edges


def get_graph_given_edges(scale, edges, rho, fscalar):

    edge_scales = edges["scales"]
    if isinstance(edge_scales, jnp.ndarray) and edge_scales.ndim > 0:
        scale_diffs = jnp.abs(scale - edge_scales)
        idx = jnp.argmin(scale_diffs)
        edge_features = edges["features"][idx]
        senders = edges["senders"][idx]
        receivers = edges["receivers"][idx]
    else:
        edge_features = edges["features"]
        senders = edges["senders"]
        receivers = edges["receivers"]

    n_node = rho.shape[0]
    n_edge = edge_features.shape[0]

    # TODO add latent feature
    node_features = jnp.stack([jnp.tile(scale, n_node), jnp.log10(rho), jnp.arcsinh(fscalar / 100)], axis=-1)

    graph = jraph.GraphsTuple(
        nodes=node_features,
        edges=edge_features,
        senders=senders,
        receivers=receivers,
        n_node=n_node,
        n_edge=n_edge,
        globals=None,
    )

    return graph


def get_graph(scale, pos, rho, fscalar, k=4, boxsize=None):
    scale = jax.lax.stop_gradient(scale)
    pos = jax.lax.stop_gradient(pos)
    rho = jax.lax.stop_gradient(rho)
    fscalar = jax.lax.stop_gradient(fscalar)

    edges = get_edges(pos, scale, k, boxsize=boxsize)
    graph = get_graph_given_edges(scale, edges, rho, fscalar)

    return graph


# def get_graph(scale, pos, rho, fscalar, k=4, boxsize=None):
#     scale = jax.lax.stop_gradient(scale)
#     pos = jax.lax.stop_gradient(pos)
#     rho = jax.lax.stop_gradient(rho)
#     fscalar = jax.lax.stop_gradient(fscalar)

#     print(scale.shape)
#     print(pos.shape)
#     print(rho.shape)
#     print(fscalar.shape)

#     n_node = pos.shape[0]
#     n_edge = k * n_node

#     k_dist, k_idx = jax_get_knn(pos, k, boxsize=boxsize)

#     node_features = jnp.stack([jnp.tile(scale, n_node), jnp.log10(rho), jnp.arcsinh(fscalar / 100)], axis=-1)
#     edge_features = k_dist.reshape(-1, 1)

#     senders = k_idx.reshape(-1)
#     receivers = jnp.repeat(jnp.arange(n_node, dtype=jnp.int32), k)

#     graph = jraph.GraphsTuple(
#         nodes=node_features,
#         edges=edge_features,
#         senders=senders,
#         receivers=receivers,
#         n_node=n_node,
#         n_edge=n_edge,
#         globals=None,
#     )

#     return graph
