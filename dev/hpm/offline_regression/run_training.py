import os, wandb, h5py
import numpy as np
from tqdm import tqdm

os.environ["JAX_PLATFORMS"] = "cpu"
import jax, optax
import jax.numpy as jnp
from flax import nnx

from jaxpm import camels, data


@nnx.jit
def train_step(model, optimizer, x, y):

    def loss_fn(model):
        y_pred = model(*x, training=True)
        return jnp.mean((y - y_pred) ** 2)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)

    return loss


@nnx.jit
def vali_loop(model, X, Y):
    losses = []
    for i in range(Y.shape[0]):
        x, y = tuple(x[i] for x in X), Y[i]

        losses.append(jnp.mean((y - model(*x)) ** 2))
    losses = jnp.mean(jnp.stack(losses))

    return losses


def train_model(
    model,
    X,
    Y,
    X_vali,
    Y_vali,
    total_steps=3_000,
    vali_every=100,
    learning_rate=1e-3,
    cosine_decay=True,
):
    optimizer = optax.chain(optax.clip_by_global_norm(1), optax.adam(learning_rate))
    optimizer = nnx.Optimizer(model, optimizer)

    if cosine_decay:
        learning_rate = optax.cosine_decay_schedule(init_value=learning_rate, decay_steps=total_steps, alpha=0.1)

    losses = []
    vali_steps = []
    vali_losses = []
    vali_loss = np.inf
    for i in (pbar := tqdm.tqdm(range(total_steps))):
        # select single snapshot
        j = np.random.choice(np.arange(Y.shape[0]))
        x, y = tuple(x[j] for x in X), Y[j]

        loss = train_step(model, optimizer, x, y)
        losses.append(loss)

        if (i % vali_every == 0) and (i != 0) or i == total_steps - 1:
            vali_steps.append(i)
            vali_loss = vali_loop(model, X_vali, Y_vali)
            vali_losses.append(vali_loss)

        pbar.set_description(f"train={loss:.4f}, vali={vali_loss:.4f}")


def train_mlp(x_labels, y_labels, eps=1e-8, include_scale=True):
    HPM = "/cluster/scratch/athomsen/CV/hpm.h5"
    with h5py.File(HPM, "r") as f:
        X_particle = f["X_particle"][:]
        Y_particle = f["Y_particle"][:]

    X, _, Y, _ = data.get_offline_regression_data(
        train_dict, x_labels=x_labels, y_labels=y_labels, standardize_input=False, include_scale=include_scale
    )
    X_vali, _, Y_vali, _ = data.get_offline_regression_data(
        test_dict, x_labels=x_labels, y_labels=y_labels, standardize_input=False, include_scale=include_scale
    )

    # scaler = StandardScaler3D().fit(Y)
    # scaler = TimeIndependentScaler3D().fit(Y)
    # Y = scaler.transform(Y)
    # Y_vali = scaler.transform(Y_vali)
    # print("with standard scaler")

    model = MLP(
        X.shape[-1],
        Y.shape[-1],
        64,
        4,
        nnx.Rngs(0),
        0.0,
    )

    train_model(
        model,
        (X,),
        Y,
        (X_vali,),
        Y_vali,
    )

    pred = model(X)
    pred_vali = model(X_vali)

    with jax.default_device(jax.devices("cpu")[0]):
        plot_preds(pred, Y, pred_vali, Y_vali)

        # pred = scaler.inverse_transform(pred)
        # pred_vali = scaler.inverse_transform(pred_vali)

        i = -1
        P_pred = normalized_paint(train_dict["gas_poss"][i], jnp.squeeze(10 ** pred[i]) - eps)
        P_pred_vali = normalized_paint(test_dict["gas_poss"][i], jnp.squeeze(10 ** pred_vali[i]) - eps)

        P_camels = normalized_paint(train_dict["gas_poss"][i], train_dict["gas_Ps"][i])
        P_camels_vali = normalized_paint(test_dict["gas_poss"][i], test_dict["gas_Ps"][i])

        plot_comparison(P_camels, P_pred, pred_label="MLP", suptitle="training set")
        plot_comparison(P_camels_vali, P_pred_vali, pred_label="MLP", suptitle="validation set")

    return model


def training():
    # Access hyperparameters from wandb.config
    config = wandb.config

    wandb_run = wandb.init(
        project="hpm",
        dir="/cluster/scratch/athomsen/wandb",
        job_type="training",
        # make sure that wandb logs to the cloud
        mode="online",
        force=True,
        config=config,
        # # additional metadata
        # tags=args.wandb_tags,
        # notes=args.wandb_notes,
    )

    # Your training logic here, using config parameters
    # For example:
    # model = create_model(learning_rate=config.learning_rate,
    #                     hidden_size=config.hidden_size)

    # After training, log metrics that your sweep will optimize
    # wandb.log({"validation_loss": val_loss})

    wandb_run.finish()


def sweep():
    sweep_config = {
        "method": "random",  # Random search method (can be 'grid', 'random', or 'bayes')
        "metric": {
            "name": "validation_loss",  # Metric to optimize
            "goal": "minimize",  # Direction for optimization (minimize or maximize)
        },
        "parameters": {
            "learning_rate": {"min": 0.0001, "max": 0.1, "distribution": "log_uniform"},
            "hidden_size": {"values": [32, 64, 128, 256]},
            "batch_size": {"values": [16, 32, 64, 128]},
            "dropout_rate": {"min": 0.0, "max": 0.5},
        },
    }

    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project="hpm")

    # Start the sweep agent
    wandb.agent(sweep_id, function=training, count=10)  # Run 10 trials


def load_hpm_data(with_particle, with_field, x_labels, y_labels):
    all_y_labels = ["P", "U", "T"]

    x_indices = [i for i, label in enumerate(all_x_labels) if label in x_labels]
    y_indices = [i for i, label in enumerate(all_y_labels) if label in y_labels]

    HPM = "/cluster/scratch/athomsen/CV/hpm.h5"
    with h5py.File(HPM, "r") as f:
        if with_particle:
            all_x_labels = ["scales", "rho", "fscalar", "vel_disp", "vel_div"]
            x_indices = [i for i, label in enumerate(all_x_labels) if label in x_labels]
            y_indices = [i for i, label in enumerate(all_y_labels) if label in y_labels]

            X_particle = f["X_particle"][..., x_indices]
            Y_particle = f["Y_particle"][..., y_indices]

        if with_field:
            all_x_labels = ["rho", "fscalar", "vel_disp", "vel_div"]
            x_indices = [i for i, label in enumerate(all_x_labels) if label in x_labels]
            y_indices = [i for i, label in enumerate(all_y_labels) if label in y_labels]

            X_field = f["X_field"][..., x_indices]
            Y_field = f["Y_field"][..., y_indices]

    return X_particle, X_field, Y_particle, Y_field


if __name__ == "__main__":
    HPM = "/cluster/scratch/athomsen/CV/hpm.h5"

with h5py.File(HPM, "r") as f:
    # print(f.keys())
    X_particle = f["X_particle"][:]

    # For normal training
    # training()

    # For sweep
    # sweep()
