import matplotlib.pyplot as plt
import jax.numpy as jnp

from jaxpm.painting import cic_paint, cic_read


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
