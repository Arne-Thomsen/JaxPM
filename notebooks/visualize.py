import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

def plot_fields(fields_dict, sum_over=None):
    """
    Plots sum projections of 3D fields along different axes,
    slicing only the first `sum_over` elements along each axis.

    Args:
    - fields: list of 3D arrays representing fields to plot
    - names: list of names for each field, used in titles
    - sum_over: number of slices to sum along each axis (default: fields[0].shape[0] // 8)
    """
    sum_over = sum_over or list(fields_dict.values())[0].shape[0] // 8
    nb_rows = len(fields_dict)
    nb_cols = 3
    fig, axes = plt.subplots(nb_rows, nb_cols, figsize=(15, 5 * nb_rows))

    def plot_subplots(proj_axis, field, row, title):
        slicing = [slice(None)] * field.ndim
        slicing[proj_axis] = slice(None, sum_over)
        slicing = tuple(slicing)

        # Sum projection over the specified axis and plot
        axes[row, proj_axis].imshow(field[slicing].sum(axis=proj_axis) + 1, 
                                    cmap='magma', extent=[0, field.shape[proj_axis], 0, field.shape[proj_axis]])
        axes[row, proj_axis].set_xlabel('Mpc/h')
        axes[row, proj_axis].set_ylabel('Mpc/h')
        axes[row, proj_axis].set_title(title)

    # Plot each field across the three axes
    for i, (name, field) in enumerate(fields_dict.items()):
        for proj_axis in range(3):
            plot_subplots(proj_axis, field, i, f"{name} projection {proj_axis}")

    plt.tight_layout()
    plt.show()