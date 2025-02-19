import jax.numpy as jnp
from jaxpm.camels import preprocess_snapshots


def get_offline_regression_data(
    snapshot_dict,
    x_labels=["rho", "fscalar", "vel_disp", "vel_div"],
    y_labels=["P", "U", "T"],
    standardize_input=True,
    include_scale=True,
    include_latent=False,
):
    scales, particle_features, field_features = preprocess_snapshots(snapshot_dict)
    n_particles = particle_features["gas_rho"].shape[-1]

    X_particle = []
    X_field = []
    for x_label in x_labels:
        if x_label in ["rho", "vel_disp"]:
            X_particle.append(jnp.log10(particle_features[f"gas_{x_label}"] + 1))
            X_field.append(jnp.log10(field_features[f"{x_label}_gas"] + 1))
        elif x_label in ["fscalar", "vel_div"]:
            X_particle.append(jnp.arcsinh(particle_features[f"gas_{x_label}"]))
            X_field.append(jnp.arcsinh(field_features[f"{x_label}_gas"] / 100))
        else:
            raise ValueError(f"Unknown x_label {x_label}")
    X_particle = jnp.stack(X_particle, axis=-1)
    X_field = jnp.stack(X_field, axis=-1)

    if standardize_input:
        X_particle = (X_particle - jnp.mean(X_particle, axis=(0, 1))) / jnp.std(X_particle, axis=(0, 1))
        X_field = (X_field - jnp.mean(X_field, axis=(0, 1, 2, 3))) / jnp.std(X_field, axis=(0, 1, 2, 3))

    if include_scale:
        X_particle = jnp.concatenate(
            [jnp.repeat(scales[..., jnp.newaxis, jnp.newaxis], n_particles, axis=-2), X_particle], axis=-1
        )

    if include_latent:
        pass

    Y_particle = []
    Y_field = []
    for y_label in y_labels:
        Y_particle.append(jnp.log10(particle_features[f"gas_{y_label}"] + 1))
        Y_field.append(jnp.log10(field_features[f"{y_label}_gas"] + 1))
    Y_particle = jnp.stack(Y_particle, axis=-1)
    Y_field = jnp.stack(Y_field, axis=-1)

    return X_particle, X_field, Y_particle, Y_field
