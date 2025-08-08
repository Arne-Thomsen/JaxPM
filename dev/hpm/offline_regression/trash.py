import os, wandb, h5py
import numpy as np

os.environ["JAX_PLATFORMS"] = "cpu"
import jax
import jax.numpy as jnp

from tqdm import tqdm

from jaxpm import camels, data


def main():
    CAMELS = "/cluster/scratch/athomsen/CV"
    parts_per_dim = 64
    mesh_per_dim = parts_per_dim
    x_labels = ["rho", "fscalar", "vel_disp", "vel_div"]
    y_labels = ["U"]

    X = []
    Y = []
    for i in tqdm(range(10)):
        # for i in tqdm(range(27)):
        camels_dict = camels.load_CV_snapshots(
            os.path.join(CAMELS, f"CV_{i}"),
            mesh_per_dim,
            parts_per_dim,
            return_hydro=True,
            # i_snapshots=[0, 1],
        )

        x, _, y, _ = data.get_offline_regression_data(
            camels_dict,
            x_labels=x_labels,
            y_labels=y_labels,
            include_scale=True,
            standardize_input=False,
            standardize_label=False,
        )
        X.append(x)
        Y.append(y)

    X = jnp.concatenate(X, axis=0)
    Y = jnp.concatenate(Y, axis=0)
    print(X.shape)

    # gas_poss = []
    # gas_vels = []
    # gas_Us = []
    # for i in tqdm(range(10)):
    #     CV = os.path.join(CAMELS, f"CV_{i}", f"parts={parts_per_dim},mesh={mesh_per_dim}.h5")

    #     with h5py.File(CV, "r") as f:
    #         gas_poss.append(f["gas_poss"][:])
    #         gas_vels.append(f["gas_vels"][:])
    #         gas_Us.append(f["gas_Us"][:])

    # gas_poss = jnp.concatenate(gas_poss, axis=0)
    # gas_vels = jnp.concatenate(gas_vels, axis=0)
    # gas_Us = jnp.concatenate(gas_Us, axis=0)


if __name__ == "__main__":
    main()
