"""Training entry point for pressure model fine-tuning on CAMELS snapshots.

Usage with YAML configuration files:
python run_training.py \
    --net-config configs/net.yaml \
    --sim-config configs/sim.yaml \
    --loss-config configs/loss.yaml \
    --use_wandb

Optionally enable Weights & Biases logging and sweeps:
python run_training.py \
    --net-config configs/net.yaml \
    --sim-config configs/sim.yaml \
    --loss-config configs/loss.yaml \
    --use_wandb \
    --sweep-net-config configs/sweep.yaml

`--net-config` defines the optimization and network hyperparameters (`n_steps`,
`d_hidden`, `n_hidden`, `kernel_size`, `activation`, `learning_rate`).
`--sim-config` provides fixed simulation settings (`parts_per_dim`,
`with_latent`, `i0`, `n_ref`).
`--loss-config` contains the `ParticleLoss` parameters. When `--use_wandb` is
active, each iteration reports `train/loss` and `train/grad_norm` to the
`JaxHPM` project in wandb (online mode). Providing `--sweep-net-config` will create
a wandb sweep exploring the network configuration only.

"""

import os, argparse, tqdm, yaml
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional

import numpy as np

import jax
import jax.numpy as jnp
from flax import nnx
import optax

import jaxpm
from jaxpm import camels, training, objectives, diagnostics
from jaxpm.painting import cic_paint, cic_read
from jaxpm.nn import MLP, ConditionedCNN
from jaxpm.objectives import ParticleLoss, FieldLoss

vcic_paint = jax.vmap(cic_paint, in_axes=(None, 0, None))
vcic_read = jax.vmap(cic_read, in_axes=(0, 0))


ACTIVATIONS = {
    "relu": jax.nn.relu,
    "swish": jax.nn.swish,
    "gelu": jax.nn.gelu,
    "tanh": jnp.tanh,
    "sigmoid": jax.nn.sigmoid,
    "softplus": jax.nn.softplus,
}


REQUIRED_HPARAM_KEYS = {"n_steps", "d_hidden", "n_hidden", "kernel_size", "activation", "learning_rate"}
REQUIRED_SIM_KEYS = {"parts_per_dim", "with_latent", "i0", "n_ref"}
REQUIRED_LOSS_KEYS = {
    "w_pos",
    "w_vel",
    "w_cls",
    "w_cross",
    "w_snapshot",
    "k_max",
    "loss_type",
    "robust_scale",
    "cutoff_quantile",
}


def _load_yaml_file(path: str) -> Dict[str, Any]:
    file_path = Path(path).expanduser()
    if not file_path.exists():
        raise FileNotFoundError(f"YAML config not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping at the root of {file_path}, got {type(data).__name__}")

    return data


def _config_get(config: Any, key: str, default: Any) -> Any:
    if hasattr(config, "get"):
        try:
            return config.get(key, default)  # type: ignore[attr-defined]
        except TypeError:
            pass

    if hasattr(config, key):
        return getattr(config, key)

    try:
        return config[key]  # type: ignore[index]
    except Exception:
        return default


def _ensure_hparams(config: Mapping[str, Any]) -> Dict[str, Any]:
    missing = REQUIRED_HPARAM_KEYS - set(config.keys())
    if missing:
        raise KeyError(f"Missing required hyperparameters in config: {', '.join(sorted(missing))}")
    return dict(config)


def _ensure_sim_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    missing = REQUIRED_SIM_KEYS - set(config.keys())
    if missing:
        raise KeyError(f"Missing required simulation parameters in config: {', '.join(sorted(missing))}")
    validated = dict(config)
    validated["with_latent"] = bool(validated["with_latent"])
    return validated


def _ensure_loss_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    missing = REQUIRED_LOSS_KEYS - set(config.keys())
    if missing:
        raise KeyError(f"Missing required loss parameters in config: {', '.join(sorted(missing))}")
    return dict(config)


def setup():
    parser = argparse.ArgumentParser()

    # paths
    parser.add_argument(
        "--BASE",
        type=str,
        default="/pscratch/sd/a/athomsen/flatiron/runs",
    )
    parser.add_argument(
        "--OUT",
        type=str,
        default="test",
    )
    parser.add_argument(
        "--CAMELS",
        type=str,
        default="/pscratch/sd/a/athomsen/flatiron/CAMELS",
    )
    parser.add_argument(
        "--CODE",
        type=str,
        default="SIMBA",
    )
    # configs
    parser.add_argument(
        "--net-config",
        dest="net_config",
        type=str,
        required=True,
        help="Path to a YAML file defining hyperparameters (n_steps, d_hidden, n_hidden, kernel_size, activation, learning_rate).",
    )
    parser.add_argument(
        "--sweep-net-config",
        dest="sweep_net_config",
        type=str,
        default=None,
        help="Optional path to a YAML file describing a wandb sweep for network hyperparameters (parameter ranges, search strategy, etc).",
    )
    parser.add_argument(
        "--sim-config",
        dest="sim_config",
        type=str,
        required=True,
        help="Path to a YAML file defining simulation parameters (parts_per_dim, with_latent, i0, n_ref).",
    )
    parser.add_argument(
        "--loss-config",
        dest="loss_config",
        type=str,
        required=True,
        help="Path to a YAML file defining ParticleLoss parameters (weights, k_max, loss options).",
    )
    # wandb
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging and allow sweeps to control hyperparameters.",
    )
    parser.add_argument(
        "--sweep-count",
        dest="sweep_count",
        type=int,
        default=None,
        help="Optional maximum number of sweep runs to execute when --sweep-net-config is provided.",
    )
    parser.add_argument(
        "--wandb_tags",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of tags to attach to the wandb run.",
    )

    args, _ = parser.parse_known_args()

    if args.sweep_net_config and not args.use_wandb:
        raise ValueError("--sweep-net-config requires --use_wandb")

    return args


def get_data(args):
    train_dict = camels.load_CV_snapshots(
        "CV_0",
        mesh_per_dim,
        parts_per_dim,
        i_snapshots=None,
        CAMELS=args.CAMELS,
        CODE=args.CODE,
    )

    vali_dict = camels.load_CV_snapshots(
        "CV_1",
        mesh_per_dim,
        parts_per_dim,
        i_snapshots=None,
        CAMELS=args.CAMELS,
        CODE=args.CODE,
    )

    return train_dict, vali_dict


if __name__ == "__main__":
    args = setup()

    config_path = Path(args.net_config).expanduser()
    base_hparams = _ensure_hparams(_load_yaml_file(str(config_path)))
    print(f"Loaded hyperparameter config from {config_path}")

    sim_config_path = Path(args.sim_config).expanduser()
    sim_config = _ensure_sim_config(_load_yaml_file(str(sim_config_path)))
    print(f"Loaded simulation config from {sim_config_path}")

    loss_config_path = Path(args.loss_config).expanduser()
    loss_config = _ensure_loss_config(_load_yaml_file(str(loss_config_path)))
    print(f"Loaded loss config from {loss_config_path}")

    sweep_net_config = None
    if args.sweep_net_config:
        sweep_net_config_path = Path(args.sweep_net_config).expanduser()
        sweep_net_config = _load_yaml_file(str(sweep_net_config_path))
        print(f"Loaded sweep config from {sweep_net_config_path}")

    OUT = os.path.join(args.BASE, args.OUT)
    os.makedirs(OUT, exist_ok=True)
    print(f"Output directory: {OUT}")

    parts_per_dim = int(sim_config["parts_per_dim"])
    mesh_per_dim = parts_per_dim
    mesh_shape = [mesh_per_dim] * 3

    i0 = int(sim_config["i0"])
    n_ref = int(sim_config["n_ref"])
    with_latent = bool(sim_config["with_latent"])

    i_ref = np.linspace(0, 33, n_ref + 1, dtype=int)[1:]

    train_dict, vali_dict = get_data(args)

    cosmo = train_dict["cosmo"]
    scales = train_dict["scales"]

    # particles
    dm_poss = train_dict["dm_poss"][i_ref]
    dm_vels = train_dict["dm_vels"][i_ref]

    gas_poss = train_dict["gas_poss"][i_ref]
    gas_vels = train_dict["gas_vels"][i_ref]

    # fields
    dm_mass = cosmo.Omega_c / (cosmo.Omega_b + cosmo.Omega_c)
    gas_mass = 1 - dm_mass

    rhos_dm = vcic_paint(jnp.zeros(mesh_shape), dm_poss, dm_mass)
    deltas_dm = rhos_dm / rhos_dm.mean() - 1

    rhos_gas = vcic_paint(jnp.zeros(mesh_shape), gas_poss, gas_mass)
    deltas_gas = rhos_gas / rhos_gas.mean() - 1

    # power spectrum
    _, cls_dm = objectives.vpower_spectrum(deltas_dm)
    _, cls_gas = objectives.vpower_spectrum(deltas_gas)

    solve_ode = training.get_ode_solver(mesh_per_dim, cosmo)
    train_step = training.get_train_step(mesh_per_dim, cosmo)

    def run_training_instance(wandb_module: Optional[Any] = None):
        config_defaults = dict(base_hparams)

        wandb_run = None
        config_source: Any

        if wandb_module is not None:
            wandb_init_kwargs: Dict[str, Any] = {
                "config": config_defaults,
                "project": "JaxHPM",
                "mode": "online",
            }
            if args.wandb_tags is not None:
                wandb_init_kwargs["tags"] = args.wandb_tags

            wandb_run = wandb_module.init(**wandb_init_kwargs)
            config_source = wandb_run.config
            wandb_module.define_metric("train/step")
            wandb_module.define_metric("train/loss", step_metric="train/step")
            wandb_module.define_metric("train/grad_norm", step_metric="train/step")
        else:
            config_source = SimpleNamespace(**config_defaults)

        activation_key = str(_config_get(config_source, "activation", config_defaults["activation"])).lower()
        d_hidden = int(_config_get(config_source, "d_hidden", config_defaults["d_hidden"]))
        n_hidden = int(_config_get(config_source, "n_hidden", config_defaults["n_hidden"]))
        kernel_size = int(_config_get(config_source, "kernel_size", config_defaults["kernel_size"]))
        learning_rate = float(_config_get(config_source, "learning_rate", config_defaults["learning_rate"]))
        n_steps = int(_config_get(config_source, "n_steps", config_defaults["n_steps"]))

        if wandb_run is not None:
            hyperparam_payload = {
                "d_hidden": d_hidden,
                "n_hidden": n_hidden,
                "kernel_size": kernel_size,
                "activation": activation_key,
                "learning_rate": learning_rate,
                "n_steps": n_steps,
            }
            wandb_run.config.update(hyperparam_payload, allow_val_change=True)
            wandb_run.summary.update({f"hyperparameters/{k}": v for k, v in hyperparam_payload.items()})

        pressure_model = ConditionedCNN(
            d_in=4 + int(with_latent),
            d_out=1 + int(with_latent),
            d_hidden=d_hidden,
            n_hidden=n_hidden,
            kernel_size=(kernel_size,) * 3,
            rngs=nnx.Rngs(0),
            norm_type="layer",
            activation=ACTIVATIONS[activation_key],
            use_residual=True,
        )

        pressure_loss_fn = ParticleLoss(
            mesh_per_dim,
            w_pos=float(loss_config["w_pos"]),
            w_vel=float(loss_config["w_vel"]),
            w_cls=float(loss_config["w_cls"]),
            w_cross=float(loss_config["w_cross"]),
            w_snapshot=float(loss_config["w_snapshot"]),
            k_max=int(loss_config["k_max"]),
            loss_type=str(loss_config["loss_type"]),
            robust_scale=float(loss_config["robust_scale"]),
            cutoff_quantile=float(loss_config["cutoff_quantile"]),
        )

        clip_norm = 1
        pressure_optimizer = nnx.ModelAndOptimizer(
            pressure_model, optax.chain(optax.clip_by_global_norm(clip_norm), optax.adam(learning_rate))
        )

        def train_step_wrapper(i0, aug_key=None):
            y0 = (dm_poss[i0], dm_vels[i0], gas_poss[i0], gas_vels[i0])
            if with_latent:
                if isinstance(pressure_model, jaxpm.nn.MLP):
                    y0 += (jnp.ones((parts_per_dim**3, 1)),)
                elif isinstance(pressure_model, jaxpm.nn.ConditionedCNN):
                    y0 += (jnp.ones(mesh_shape + [1]),)
                else:
                    raise NotImplementedError
            t0 = scales[i0]

            loss, grad = train_step(
                pressure_loss_fn,
                pressure_optimizer,
                y0,
                t0,
                ref_t=scales[i_ref],
                ref_poss=gas_poss,
                ref_vels=gas_vels,
                ref_cls=cls_gas,
                ref_deltas=deltas_gas,
                pressure_model=pressure_model,
                gravity_model=None,
                model_to_train="pressure",
                aug_key=aug_key,
                tstep=scales[(t0 <= scales) & (scales <= scales[i_ref[-1]])],
                nt=2,
            )

            return loss, grad

        key = jax.random.key(71)
        for i in (pbar := tqdm.tqdm(range(n_steps))):
            key, subkey = jax.random.split(key)
            loss_value, grad_value = train_step_wrapper(i0=i0, aug_key=subkey)

            pbar.set_description(f"Step {i}")
            pbar.set_postfix({"loss": f"{loss_value:.4e}", "grad": f"{grad_value:.4e}"})

            if wandb_module is not None:
                wandb_module.log({"train/step": i, "train/loss": loss_value, "train/grad_norm": grad_value})

        if wandb_module is not None:
            wandb_module.finish()

        diagnostics.run_simulations(
            vali_dict,
            mesh_per_dim,
            # gravity_model=gravity_model,
            pressure_model=pressure_model,
            i_init=i0,
            i_plot=np.linspace(i0, 33, 4, dtype=int),
            nt=2,
            plot_dm=True,
            plot_gas=True,
            with_latent=with_latent,
            plot_latent=with_latent,
            loss_fn=pressure_loss_fn,
        )

    if args.use_wandb:
        import wandb as wandb_module

        running_inside_agent = "WANDB_SWEEP_ID" in os.environ

        if sweep_net_config is not None and not running_inside_agent:
            sweep_kwargs: Dict[str, Any] = {"project": "JaxHPM"}

            sweep_id = wandb_module.sweep(sweep_net_config, **sweep_kwargs)
            print(f"Initialized wandb sweep: {sweep_id}")

            wandb_module.agent(
                sweep_id,
                function=lambda: run_training_instance(wandb_module),
                count=args.sweep_count,
            )
        else:
            run_training_instance(wandb_module)
    else:
        run_training_instance(None)
