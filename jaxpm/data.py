import jax
import jax.numpy as jnp

from jaxpm.painting import cic_paint, cic_read
from jaxpm.camels import preprocess_snapshots
from jaxpm.kernels import fftk, invnabla_kernel, invlaplace_kernel, gradient_kernel


def get_offline_regression_data(
    snapshot_dict,
    x_labels=["rho", "fscalar", "vel_disp", "vel_div", "tidal_eigval"],
    y_labels=["P", "U", "T"],
    include_scale=True,
    include_latent=False,
    standardize_input=False,
    standardize_label=False,
    eps=1e-8,
):
    scales, particle_features, field_features = preprocess_snapshots(snapshot_dict)
    n_particles = particle_features["gas_rho"].shape[-1]

    X_particle = []
    X_field = []
    for x_label in x_labels:
        if x_label in ["rho", "vel_disp"]:
            X_particle.append(jnp.log10(particle_features[f"gas_{x_label}"] + eps))
            X_field.append(jnp.log10(field_features[f"{x_label}_gas"] + eps))
        elif x_label in ["fscalar", "vel_div"] or "tidal_eigval" in x_label:
            X_particle.append(jnp.arcsinh(particle_features[f"gas_{x_label}"]))
            X_field.append(jnp.arcsinh(field_features[f"{x_label}_gas"]))
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
        Y_particle.append(jnp.log10(particle_features[f"gas_{y_label}"] + eps))
        Y_field.append(jnp.log10(field_features[f"{y_label}_gas"] + eps))
    Y_particle = jnp.stack(Y_particle, axis=-1)
    Y_field = jnp.stack(Y_field, axis=-1)

    if standardize_label:
        Y_particle = (Y_particle - jnp.mean(Y_particle, axis=(0, 1))) / jnp.std(Y_particle, axis=(0, 1))
        Y_field = (Y_field - jnp.mean(Y_field, axis=(0, 1, 2, 3))) / jnp.std(Y_field, axis=(0, 1, 2, 3))

    return X_particle, X_field, Y_particle, Y_field


def get_hpm_inputs(
    scale,
    gas_pos,
    gas_vel,
    gas_rho,
    rho_gas,
    gas_N,
    mesh_shape,
    gas_latent=None,
    latent_gas=None,
    return_vel=True,
    return_field=False,
    eps=1e-8,
):
    # fscalar
    kvec = fftk(mesh_shape)
    rho_gas_k = jnp.fft.rfftn(rho_gas)
    fscalar_gas = jnp.fft.irfftn(rho_gas_k * invnabla_kernel(kvec))
    gas_fscalar = cic_read(fscalar_gas, gas_pos)

    gas_inputs = [
        jnp.tile(scale, gas_pos.shape[0]),
        jnp.log10(gas_rho + eps),
        jnp.arcsinh(gas_fscalar),
    ]
    if return_field:
        field_inputs = [
            jnp.log10(rho_gas + eps),
            jnp.arcsinh(fscalar_gas),
        ]

    if return_vel:
        # velocity dispersion
        vcic_paint = jax.vmap(cic_paint, in_axes=(None, None, -1), out_axes=-1)
        vcic_read = jax.vmap(cic_read, in_axes=(-1, None), out_axes=-1)
        vel_mean_gas = vcic_paint(jnp.zeros(mesh_shape), gas_pos, gas_vel / gas_N[..., jnp.newaxis])
        gas_vel_mean = vcic_read(vel_mean_gas, gas_pos)
        gas_vel_disp = jnp.sum((gas_vel_mean - gas_vel) ** 2, axis=-1)

        # velocity divergence
        vel_gas_k = jnp.fft.rfftn(vel_mean_gas, axes=(0, 1, 2))
        gas_vel_div = [
            cic_read(jnp.fft.irfftn(gradient_kernel(kvec, i) * vel_gas_k[..., i], axes=(0, 1, 2)), gas_pos)
            for i in range(len(kvec))
        ]
        gas_vel_div = jnp.stack(gas_vel_div, axis=-1)
        gas_vel_div = jnp.sum(gas_vel_div, axis=-1)

        gas_inputs += [
            jnp.log10(gas_vel_disp + eps),
            jnp.arcsinh(gas_vel_div),
        ]

        if return_field:
            vel_disp_gas = cic_paint(jnp.zeros(mesh_shape), gas_pos, gas_vel_disp / gas_N)
            vel_div_gas = cic_paint(jnp.zeros(mesh_shape), gas_pos, gas_vel_div / gas_N)
            field_inputs += [
                jnp.log10(vel_disp_gas + eps),
                jnp.arcsinh(vel_div_gas),
            ]

    # these are all scalar features and can be stacked
    gas_inputs = jnp.stack(gas_inputs, axis=-1)
    if return_field:
        field_inputs = jnp.stack(field_inputs, axis=-1)

    # latent might be higher dimensional
    if gas_latent is not None:
        assert gas_latent.ndim == 2

        # latent_gas = vcic_paint(jnp.zeros(mesh_shape), gas_pos, gas_latent / gas_N[..., jnp.newaxis])
        # latent_gas_k = jnp.fft.rfftn(latent_gas, axes=(0, 1, 2))
        # latent_kernel_gas_k = latent_gas_k * invnabla_kernel(kvec)[..., jnp.newaxis]
        # # latent_kernel_gas_k = latent_gas_k * invlaplace_kernel(kvec)[..., jnp.newaxis]
        # latent_kernel_gas = jnp.fft.irfftn(latent_kernel_gas_k, axes=(0, 1, 2))
        # gas_latent_kernel = vcic_read(latent_kernel_gas, gas_pos)

        # gas_inputs = jnp.concatenate([gas_inputs, gas_latent, gas_latent_kernel], axis=-1)

        gas_inputs = jnp.concatenate([gas_inputs, gas_latent], axis=-1)

        # TODO
        if return_field:
            latent_gas = vcic_paint(jnp.zeros(mesh_shape), gas_pos, gas_latent / gas_N[..., jnp.newaxis])
            field_inputs = jnp.concatenate([field_inputs, latent_gas], axis=-1)

    elif latent_gas is not None:
        field_inputs = jnp.concatenate([field_inputs, latent_gas], axis=-1)

    if return_field:
        return gas_inputs, field_inputs
    else:
        return gas_inputs


def get_simplified_hpm_inputs(
    scale,
    gas_pos,
    gas_vel,
    mesh_shape,
    gas_latent=None,
    return_vel=True,
    eps=1e-8,
):
    N_gas = cic_paint(jnp.zeros(mesh_shape), gas_pos)
    gas_N = cic_read(N_gas, gas_pos)

    # TODO weight doesn't matter
    rho_gas = N_gas
    gas_rho = gas_N

    # fscalar
    kvec = fftk(mesh_shape)
    rho_gas_k = jnp.fft.rfftn(rho_gas)
    fscalar_gas = jnp.fft.irfftn(rho_gas_k * invnabla_kernel(kvec))
    gas_fscalar = cic_read(fscalar_gas, gas_pos)

    gas_inputs = [
        jnp.tile(scale, gas_pos.shape[0]),
        jnp.log10(gas_rho + eps),
        jnp.arcsinh(gas_fscalar),
    ]

    if return_vel:
        # velocity dispersion
        vcic_paint = jax.vmap(cic_paint, in_axes=(None, None, -1), out_axes=-1)
        vcic_read = jax.vmap(cic_read, in_axes=(-1, None), out_axes=-1)
        vel_mean_gas = vcic_paint(jnp.zeros(mesh_shape), gas_pos, gas_vel / gas_N[..., jnp.newaxis])
        gas_vel_mean = vcic_read(vel_mean_gas, gas_pos)
        gas_vel_disp = jnp.sum((gas_vel_mean - gas_vel) ** 2, axis=-1)

        # velocity divergence
        vel_gas_k = jnp.fft.rfftn(vel_mean_gas, axes=(0, 1, 2))
        gas_vel_div = [
            cic_read(jnp.fft.irfftn(gradient_kernel(kvec, i) * vel_gas_k[..., i], axes=(0, 1, 2)), gas_pos)
            for i in range(len(kvec))
        ]
        gas_vel_div = jnp.stack(gas_vel_div, axis=-1)
        gas_vel_div = jnp.sum(gas_vel_div, axis=-1)

        gas_inputs += [
            jnp.log10(gas_vel_disp + eps),
            jnp.arcsinh(gas_vel_div),
        ]

    # these are all scalar features and can be stacked
    gas_inputs = jnp.stack(gas_inputs, axis=-1)

    # latent might be higher dimensional
    if gas_latent is not None:
        assert gas_latent.ndim == 2
        gas_inputs = jnp.concatenate([gas_inputs, gas_latent], axis=-1)

    return gas_inputs
