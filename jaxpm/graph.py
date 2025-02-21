import numpy as np
import scipy.spatial

import jax
import jax.numpy as jnp
import jraph

from functools import partial

from jaxpm import data


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


def get_graph_given_edges(node_features, edges, current_scale=None):
    interpolate_graph = True if current_scale is not None else False

    # single graph
    if node_features.ndim == 2:
        # node_features is single snapshot, but edges includes multiple. Choose the closest one to current_scale
        if interpolate_graph:
            scale_diffs = jnp.abs(current_scale - edges["scales"])
            idx = jnp.argmin(scale_diffs)
            edge_features = edges["features"][idx]
            senders = edges["senders"][idx]
            receivers = edges["receivers"][idx]
        else:
            edge_features = edges["features"]
            senders = edges["senders"]
            receivers = edges["receivers"]

        graph = jraph.GraphsTuple(
            nodes=node_features,
            edges=edge_features,
            senders=senders,
            receivers=receivers,
            n_node=node_features.shape[0],
            n_edge=edge_features.shape[0],
            globals=None,
        )

    # multiple graphs (one per scale)
    elif node_features.ndim == 3:
        n_scales = node_features.shape[0]

        graph = []
        for i in range(n_scales):
            graph.append(
                jraph.GraphsTuple(
                    nodes=node_features[i],
                    edges=edges["features"][i],
                    senders=edges["senders"][i],
                    receivers=edges["receivers"][i],
                    n_node=node_features.shape[1],
                    n_edge=edges["features"].shape[1],
                    globals=None,
                )
            )

    return graph


def get_graphs_from_snapshots(
    snapshot_dict,
    x_labels=["rho", "fscalar", "vel_disp", "vel_div"],
    y_labels=["P", "U", "T"],
    k=4,
    boxsize=None,
):
    """For offline regression"""

    node_features, _, Y_particle, _ = data.get_offline_regression_data(
        snapshot_dict, x_labels=x_labels, y_labels=y_labels
    )

    edges = get_edges(snapshot_dict["gas_poss"], snapshot_dict["scales"], k=k, boxsize=boxsize)
    graph = get_graph_given_edges(node_features, edges)

    return graph, Y_particle


def get_graph_from_features(node_features, scale, k=4, boxsize=None, stop_gradient=True):
    """For online/in-sim learning"""

    if stop_gradient:
        node_features = jax.lax.stop_gradient(node_features)

    edges = get_edges(node_features, scale, k=k, boxsize=boxsize)
    graph = get_graph_given_edges(node_features, edges)

    return graph
