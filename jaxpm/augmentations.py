import jax
import jax.numpy as jnp


def _get_rotation_matrix(rot_z, rot_x, rot_y):
    """
    Compute the combined 3D rotation matrix from three sequential 90-degree rotations.

    Args:
        rot_z: Rotation around z-axis (0-3, corresponding to 0°, 90°, 180°, 270°)
        rot_x: Rotation around x-axis (0-3, corresponding to 0°, 90°, 180°, 270°)
        rot_y: Rotation around y-axis (0-3, corresponding to 0°, 90°, 180°, 270°)

    Returns:
        Combined rotation matrix [3, 3]
    """

    # Around z-axis: rotates xy plane
    rot_z_matrices = jnp.array(
        [
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # k=0: identity
            [[0, -1, 0], [1, 0, 0], [0, 0, 1]],  # k=1: 90°
            [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],  # k=2: 180°
            [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],  # k=3: 270°
        ]
    )

    # Around x-axis: rotates yz plane
    rot_x_matrices = jnp.array(
        [
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # k=0: identity
            [[1, 0, 0], [0, 0, -1], [0, 1, 0]],  # k=1: 90°
            [[1, 0, 0], [0, -1, 0], [0, 0, -1]],  # k=2: 180°
            [[1, 0, 0], [0, 0, 1], [0, -1, 0]],  # k=3: 270°
        ]
    )

    # Around y-axis: rotates xz plane
    rot_y_matrices = jnp.array(
        [
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # k=0: identity
            [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],  # k=1: 90°
            [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],  # k=2: 180°
            [[0, 0, -1], [0, 1, 0], [1, 0, 0]],  # k=3: 270°
        ]
    )

    R_z, R_x, R_y = rot_z_matrices[rot_z], rot_x_matrices[rot_x], rot_y_matrices[rot_y]
    R = R_y @ R_x @ R_z

    return R


def _apply_field_transform(field, rot_z, rot_x, rot_y, flip_x, flip_y, flip_z):
    """
    Helper function to apply rotation and flip transformations to a 3D or 4D field.

    The transformation mirrors the matrix-based particle transform so that
    sampled field values remain aligned with rotated/flipped particle
    positions.

    Args:
        field: 3D or 4D array to transform
               - 3D: (nx, ny, nz)
               - 4D: (n_time, nx, ny, nz)
        rot_z, rot_x, rot_y: rotation parameters (0-3) for each axis
        flip_x, flip_y, flip_z: boolean flip parameters for each axis

    Returns:
        Transformed field with same shape as input
    """

    offset = field.ndim - 3
    spatial_shape = field.shape[offset:]

    coords = jnp.moveaxis(jnp.indices(spatial_shape, dtype=jnp.int32), 0, -1)
    coords = coords.reshape(-1, 3)

    mesh = jnp.asarray(spatial_shape, dtype=jnp.int32).reshape(1, 3)
    flips = jnp.asarray([flip_x, flip_y, flip_z]).reshape(1, 3)

    # Undo flips to map back to original grid coordinates
    coords = jnp.where(flips, jnp.mod(mesh - coords, mesh), coords)

    # Undo rotations using the forward rotation matrix transpose relationship
    R = _get_rotation_matrix(rot_z, rot_x, rot_y).astype(jnp.int32)
    coords = coords @ R
    coords = jnp.mod(coords, mesh)

    coords = coords.reshape(spatial_shape + (3,))
    xi = coords[..., 0]
    yi = coords[..., 1]
    zi = coords[..., 2]

    index_prefix = (slice(None),) * offset
    transformed = field[index_prefix + (xi, yi, zi)]

    return transformed


def _apply_position_transform(pos, mesh_per_dim, rot_z, rot_x, rot_y, flip_x, flip_y, flip_z):
    """
    Helper function to apply rotation and flip transformations to particle positions.

    Args:
        pos: particle positions [..., 3] - can handle (n_time, n_particles, 3) or (n_particles, 3)
        mesh_per_dim: size of the periodic box in each dimension
        rot_z, rot_x, rot_y: rotation parameters (0-3) for each axis
        flip_x, flip_y, flip_z: boolean flip parameters for each axis

    Returns:
        Transformed positions with same shape as input
    """
    R = _get_rotation_matrix(rot_z, rot_x, rot_y)

    # Apply rotation: [..., 3] @ [3, 3] -> [..., 3]
    pos_rot = pos @ R.T

    # Handle periodicity for rotated positions
    pos_rot = pos_rot % mesh_per_dim

    # Apply flips
    flips = jnp.array([flip_x, flip_y, flip_z])
    pos_final = jnp.where(flips, (mesh_per_dim - pos_rot) % mesh_per_dim, pos_rot)

    return pos_final


def _apply_velocity_transform(vel, rot_z, rot_x, rot_y, flip_x, flip_y, flip_z):
    """
    Helper function to apply rotation and flip transformations to particle velocities.

    Args:
        vel: particle velocities [..., 3] - same shape as positions
        rot_z, rot_x, rot_y: rotation parameters (0-3) for each axis
        flip_x, flip_y, flip_z: boolean flip parameters for each axis

    Returns:
        Transformed velocities with same shape as input
    """
    R = _get_rotation_matrix(rot_z, rot_x, rot_y)

    # Apply rotation: [..., 3] @ [3, 3] -> [..., 3]
    vel_rot = vel @ R.T

    # Apply flips (velocities get sign flip)
    flips = jnp.array([flip_x, flip_y, flip_z])
    vel_final = jnp.where(flips, -vel_rot, vel_rot)

    return vel_final


def rot_flip_3d(key, mesh_per_dim, field=None, pos=None, vel=None):
    """
    Unified function to apply consistent random 90-degree rotations and flips
    to 3D fields and/or particle positions and velocities.

    This function generates a single set of random transformation parameters
    and applies them consistently across all provided inputs, ensuring that
    field and particle data remain properly aligned.

    Args:
        key: JAX random key for generating transformation parameters
        mesh_per_dim: Size of the periodic box in each dimension
        field: Optional 3D or 4D array to transform - can handle (nx, ny, nz) or (n_time, nx, ny, nz)
        pos: Optional particle positions [..., 3] - can handle (n_time, n_particles, 3) or (n_particles, 3)
        vel: Optional particle velocities [..., 3] - same shape as pos

    Returns:
        Dictionary with transformed inputs

    Raises:
        ValueError: If pos/vel are provided without mesh_per_dim, or if pos and vel have different shapes
    """

    key1, key2, key3, key4, key5, key6 = jax.random.split(key, 6)

    rot_x = jax.random.randint(key1, (), 0, 4)
    rot_y = jax.random.randint(key2, (), 0, 4)
    rot_z = jax.random.randint(key3, (), 0, 4)

    flip_x = jax.random.bernoulli(key4)
    flip_y = jax.random.bernoulli(key5)
    flip_z = jax.random.bernoulli(key6)

    result = {}
    if field is not None:
        result["field"] = _apply_field_transform(field, rot_z, rot_x, rot_y, flip_x, flip_y, flip_z)
    if pos is not None:
        result["pos"] = _apply_position_transform(pos, mesh_per_dim, rot_z, rot_x, rot_y, flip_x, flip_y, flip_z)
    if vel is not None:
        result["vel"] = _apply_velocity_transform(vel, rot_z, rot_x, rot_y, flip_x, flip_y, flip_z)

    return result
