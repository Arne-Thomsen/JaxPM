import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from functools import partial
from tqdm import tqdm

import jax
import jax.numpy as jnp

from jaxpm.painting import cic_paint, cic_read, compensate_cic
from jaxpm.utils import power_spectrum, cross_correlation_coefficients


def plot_particle_evolution(
    mesh_shape,
    scales,
    positions,
    weights=None,
    # values
    log=True,
    vmin=None,
    vmax=None,
    shared_colorbar=True,
    individual_colorbars=False,
    return_lims=False,
    # cosmetics
    ncols=4,
    title=None,
    cmap="magma",
):
    assert 3 == len(mesh_shape) == positions[0].shape[-1]
    assert not (shared_colorbar and individual_colorbars)

    nrows = len(scales) // ncols
    weights = weights if weights is not None else [None] * len(scales)

    fields = []
    i = 0
    for j in range(nrows):
        for k in range(ncols):
            field = cic_paint(jnp.zeros(mesh_shape), positions[i], weights[i])
            field = field.sum(axis=0)
            if log:
                field = jnp.log10(field)

            i += 1
            fields.append(field)
    fields = jnp.stack(fields)

    if shared_colorbar:
        vmin = vmin if vmin is not None else fields.min()
        vmax = vmax if vmax is not None else fields.max()

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(2 * ncols, 2 * nrows), constrained_layout=True)

    i = 0
    for j in tqdm(range(nrows)):
        for k in range(ncols):
            if individual_colorbars:
                vmin = vmin if vmin is not None else fields[0].min()
                vmax = vmax if vmax is not None else fields[0].max()

            im = ax[j, k].imshow(
                fields[i],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )

            ax[j, k].set_title(f"{scales[i]:.4f}", fontsize=12)
            ax[j, k].set_xticks([])
            ax[j, k].set_yticks([])

            if individual_colorbars:
                fig.colorbar(im, ax=ax[j, k], orientation="horizontal", shrink=0.6, aspect=20)

            i += 1

    if shared_colorbar:
        fig.colorbar(im, ax=ax[:, :], orientation="horizontal", shrink=0.8, aspect=50)

    if title is not None:
        fig.suptitle(title, fontsize=16, y=1.05)

    if return_lims:
        return vmin, vmax


def compare_particle_evolution(
    mesh_shape,
    scales,
    positions,
    weights=None,
    include_pk=False,
    include_reference=False,
    # values
    log=True,
    vmin=None,
    vmax=None,
    shared_colorbar=True,
    individual_colorbars=False,
    # cosmetics
    title=None,
    col_titles=None,
    cmap="magma",
):
    assert 3 == len(mesh_shape) == positions[0].shape[-1]
    assert not (shared_colorbar and individual_colorbars)
    assert positions.ndim == 4

    n_scales = len(scales)
    n_runs = len(positions)
    colors = sns.color_palette("tab10", n_colors=n_runs)

    if weights is None:
        vcic_paint = jax.vmap(jax.vmap(cic_paint, in_axes=(None, 0, None)), in_axes=(None, 0, None))
    else:
        vcic_paint = jax.vmap(jax.vmap(cic_paint, in_axes=(None, 0, 0)), in_axes=(None, 0, 0))
    fields = vcic_paint(jnp.zeros(mesh_shape), positions, weights)
    fields_2d = fields.sum(axis=2)
    if log:
        fields_2d = jnp.log10(fields_2d)

    delta_fields = fields_2d[0] - fields_2d[1:]

    if shared_colorbar:
        vmin = vmin if vmin is not None else fields_2d.min()
        vmax = vmax if vmax is not None else fields_2d.max()

    nrows = n_scales
    ncols = n_runs
    if include_pk:
        ncols += 1
    if include_reference:
        ncols += 2 + n_runs - 1

    fig, ax = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(2 * ncols, 2 * nrows), constrained_layout=True, sharey="col", sharex="col"
    )

    for i in tqdm(range(n_scales)):
        for j in range(n_runs):
            label = col_titles[j] if col_titles is not None else None

            if individual_colorbars:
                vmin = vmin if vmin is not None else fields_2d[0].min()
                vmax = vmax if vmax is not None else fields_2d[0].max()

            im = ax[i, j].imshow(
                fields_2d[j, i],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )

            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])

            if individual_colorbars:
                fig.colorbar(im, ax=ax[i, j], orientation="horizontal", shrink=0.6, aspect=20)

            @jax.jit
            def jit_power_spectrum(field):
                return power_spectrum(
                    compensate_cic(field),
                    boxsize=np.array([25.0] * 3),
                    kmin=np.pi / 25.0,
                    dk=2 * np.pi / 25.0,
                )

            @jax.jit
            def jit_cross_correlation(field_a, field_b):
                return cross_correlation_coefficients(
                    compensate_cic(field_a),
                    compensate_cic(field_b),
                    boxsize=np.array([25.0] * 3),
                    kmin=np.pi / 25.0,
                    dk=2 * np.pi / 25.0,
                )

            if include_pk:
                axis = n_runs
                k, pk = jit_power_spectrum(fields[j, i])
                ax[i, axis].loglog(k, pk, label=label)
                ax[i, axis].set(ylabel=r"$P(k)$")
                ax[0, axis].legend()
                ax[0, axis].set(title="power spectrum")
                ax[n_scales - 1, axis].set(xlabel=r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")

            if include_reference and j != 0:
                # residual map
                axis = n_runs + j
                im_delta = ax[i, axis].imshow(
                    delta_fields[j - 1, i],
                    cmap=cmap,
                    vmin=delta_fields.min(),
                    vmax=delta_fields.max(),
                )
                ax[i, axis].set_xticks([])
                ax[i, axis].set_yticks([])

                # normalized power spectrum
                _, pk0 = jit_power_spectrum(fields[0, i])
                ax[i, -2].axhline(0.0, color=colors[0], linestyle="--")
                ax[i, -2].plot(k, pk / pk0 - 1, label=label, color=colors[j])
                ax[i, -2].set(xscale="log")
                ax[i, -2].set(ylabel=r"$P(k)/P_\text{ref} - 1$")

                # normalized cross-correlation
                k, pck = jit_cross_correlation(fields[0, i], fields[j, i])
                ax[i, -1].axhline(1.0, color=colors[0], linestyle="--")
                ax[i, -1].plot(k, pck / jnp.sqrt(pk * pk0), label=label, color=colors[j])
                ax[i, -1].set(xscale="log")
                ax[i, -1].set(ylabel=r"$P_\text{cross}(k)/\sqrt{P_\text{ref}(k) P(k)}$")

                if i == 0:
                    ax[i, -2].set(title="normalized\n power spectrum")
                    ax[i, -1].set(title="normalized\n cross-correlation")
                if i == n_scales - 1:
                    ax[i, -2].set(xlabel=r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
                    ax[i, -1].set(xlabel=r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")

    for i, scale in enumerate(scales):
        ax[i, 0].set_ylabel(f"{scale:.4f}", fontsize=12)

    if col_titles is not None:
        for j, col_title in enumerate(col_titles):
            ax[0, j].set_title(col_title, fontsize=12)
            if include_reference and j != 0:
                ax[0, n_runs + j].set_title("residual\n" + col_title, fontsize=12)

    if shared_colorbar:
        fig.colorbar(im, ax=ax[:, :n_runs], orientation="horizontal", shrink=0.8, aspect=20, label="log(sum(field))")

    if include_reference:
        fig.colorbar(
            im_delta,
            ax=ax[:, (n_runs + 1) : ((n_runs + 1) + (n_runs - 1))],
            orientation="horizontal",
            shrink=0.8,
            aspect=20,
            label="log(sum(field_a)) - log(sum(field_b))",
        )

    if title is not None:
        fig.suptitle(title, fontsize=16, y=1.05)
