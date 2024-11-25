import matplotlib.pyplot as plt
import numpy as np

import jax
import jax.numpy as jnp

from jaxpm.painting import cic_paint, cic_read, compensate_cic
from jaxpm.utils import power_spectrum


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
    for j in range(nrows):
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

    if weights is None:
        vcic_paint = jax.vmap(jax.vmap(cic_paint, in_axes=(None, 0, None)), in_axes=(None, 0, None))
    else:
        vcic_paint = jax.vmap(jax.vmap(cic_paint, in_axes=(None, 0, 0)), in_axes=(None, 0, 0))
    fields = vcic_paint(jnp.zeros(mesh_shape), positions, weights)
    fields_2d = fields.sum(axis=2)
    if log:
        fields_2d = jnp.log10(fields_2d)

    if shared_colorbar:
        vmin = vmin if vmin is not None else fields_2d.min()
        vmax = vmax if vmax is not None else fields_2d.max()

    nrows = len(scales)
    ncols = len(positions)
    if include_pk:
        ncols += 1

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(2 * ncols, 2 * nrows), constrained_layout=True)

    for i in range(len(scales)):
        for j in range(len(positions)):
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

            if include_pk:
                k, pk = power_spectrum(
                    compensate_cic(np.asarray(fields[j, i])),
                    boxsize=np.array([25.0] * 3),
                    kmin=np.pi / 25.0,
                    dk=2 * np.pi / 25.0,
                )
                ax[i, -1].loglog(k, pk, label=label)

        if include_pk:
            ax[0, -1].legend()
            ax[i, -1].set(xlabel=r"$k$ [$h \ \mathrm{Mpc}^{-1}$]", ylabel=r"$P(k)$")

    for i, scale in enumerate(scales):
        ax[i, 0].set_ylabel(f"{scale:.4f}", fontsize=12)

    if col_titles is not None:
        for j, col_title in enumerate(col_titles):
            ax[0, j].set_title(col_title, fontsize=12)

    if shared_colorbar:
        fig.colorbar(im, ax=ax[:, :], orientation="horizontal", shrink=0.8, aspect=20)

    if title is not None:
        fig.suptitle(title, fontsize=16, y=1.05)
