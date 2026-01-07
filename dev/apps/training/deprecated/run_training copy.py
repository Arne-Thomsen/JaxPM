"""Training entry point for pressure model fine-tuning on CAMELS snapshots.

Usage with a YAML hyperparameter config:
python run_training.py \
    --config configs/train.yaml \
    --n_steps 100

Optionally enable Weights & Biases logging and sweeps:
python run_training.py \
    --config configs/train.yaml \
    --use_wandb \
    --sweep-config configs/sweep.yaml

`--config` should define the scalar hyperparameters (`d_hidden`, `n_hidden`,
`kernel_size`, `activation`, `learning_rate`). When `--use_wandb` is active,
each training iteration reports `train/loss` and `train/grad_norm` to the
`JaxHPM` project in wandb (online mode). Providing `--sweep-config` will create
a wandb sweep using the ranges in that YAML file.

"""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import tqdm
import yaml

import numpy as np

import jax
import jax.numpy as jnp
from flax import nnx
import optax

import jaxpm
from jaxpm import camels, training, objectives
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


REQUIRED_HPARAM_KEYS = {"d_hidden", "n_hidden", "kernel_size", "activation", "learning_rate"}


def load_yaml_file(path: str) -> Dict[str, Any]:
    file_path = Path(path).expanduser()
    if not file_path.exists():
        raise FileNotFoundError(f"YAML config not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping at the root of {file_path}, got {type(data).__name__}")

    return data


def config_get(config: Any, key: str, default: Any) -> Any:
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


def ensure_hparams(config: Mapping[str, Any]) -> Dict[str, Any]:
    missing = REQUIRED_HPARAM_KEYS - set(config.keys())
    if missing:
        raise KeyError(f"Missing required hyperparameters in config: {', '.join(sorted(missing))}")
    return dict(config)


@dataclass(frozen=True)
class HyperParams:
    d_hidden: int
    n_hidden: int
    kernel_size: int
    activation: str
    learning_rate: float

    @classmethod
    def from_config(cls, source: Any, defaults: Mapping[str, Any]) -> "HyperParams":
        def fetch(key: str) -> Any:
            return config_get(source, key, defaults[key])

        activation = str(fetch("activation")).lower()
        if activation not in ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{activation}'. Available options: {', '.join(sorted(ACTIVATIONS.keys()))}"
            )

        try:
            kernel_size = int(fetch("kernel_size"))
        except (TypeError, ValueError) as exc:
            raise TypeError("kernel_size must be an integer") from exc
        if kernel_size <= 0:
            raise ValueError("kernel_size must be a positive integer")

        return cls(
            d_hidden=int(fetch("d_hidden")),
            n_hidden=int(fetch("n_hidden")),
            kernel_size=kernel_size,
            activation=activation,
            learning_rate=float(fetch("learning_rate")),
        )

    @property
    def activation_fn(self):
        return ACTIVATIONS[self.activation]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "d_hidden": self.d_hidden,
            "n_hidden": self.n_hidden,
            "kernel_size": self.kernel_size,
            "activation": self.activation,
            "learning_rate": self.learning_rate,
        }


@dataclass(frozen=True)
class TrainingData:
    mesh_shape: tuple[int, int, int]
    mesh_per_dim: int
    parts_per_dim: int
    i0: int
    i_ref: np.ndarray
    scales: jnp.ndarray
    ref_scales: jnp.ndarray
    ref_stop: float
    dm_poss: jnp.ndarray
    dm_vels: jnp.ndarray
    gas_poss: jnp.ndarray
    gas_vels: jnp.ndarray
    cls_gas: jnp.ndarray
    deltas_gas: jnp.ndarray
    train_step: Any


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments used to configure the training run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--BASE", type=str, default="/pscratch/sd/a/athomsen/flatiron/runs")
    parser.add_argument("--OUT", type=str, default="test")
    parser.add_argument("--CAMELS", type=str, default="/pscratch/sd/a/athomsen/flatiron/CAMELS")
    parser.add_argument("--CODE", type=str, default="SIMBA")
    parser.add_argument("--parts_per_dim", type=int, default=64)
    parser.add_argument("--with_latent", action="store_true")
    parser.add_argument("--i0", type=int, default=0)
    parser.add_argument("--n_ref", type=int, default=4)
    parser.add_argument("--n_steps", type=int, default=100)
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a YAML file defining hyperparameters (d_hidden, n_hidden, kernel_size, activation, learning_rate).",
    )
    parser.add_argument(
        "--sweep-config",
        dest="sweep_config",
        type=str,
        default=None,
        help="Optional path to a YAML file describing a wandb sweep (parameter ranges, search strategy, etc).",
    )
    parser.add_argument(
        "--sweep-count",
        dest="sweep_count",
        type=int,
        default=None,
        help="Optional maximum number of sweep runs to execute when --sweep-config is provided.",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging and allow sweeps to control hyperparameters.",
    )
    parser.add_argument(
        "--wandb_tags",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of tags to attach to the wandb run.",
    )

    args, _ = parser.parse_known_args()
    return args


def load_configurations(args: argparse.Namespace) -> tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Load base hyperparameters and optional wandb sweep configuration from disk."""
    config_path = Path(args.config).expanduser()
    base_hparams = ensure_hparams(load_yaml_file(str(config_path)))
    print(f"Loaded hyperparameter config from {config_path}")

    sweep_config = None
    if args.sweep_config:
        sweep_path = Path(args.sweep_config).expanduser()
        sweep_config = load_yaml_file(str(sweep_path))
        print(f"Loaded sweep config from {sweep_path}")

    return base_hparams, sweep_config


def prepare_output_directory(base_path: str, run_name: str) -> str:
    """Create (or reuse) the output directory and report its location."""
    output_dir = os.path.join(base_path, run_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    return output_dir


def build_training_data(args: argparse.Namespace) -> TrainingData:
    """Assemble the training arrays and metadata required for optimisation."""
    parts_per_dim = args.parts_per_dim
    mesh_per_dim = parts_per_dim
    mesh_shape = (mesh_per_dim,) * 3
    i_ref = np.linspace(0, 33, args.n_ref + 1, dtype=int)[1:]

    train_dict = camels.load_CV_snapshots(
        "CV_0",
        mesh_per_dim,
        parts_per_dim,
        i_snapshots=None,
        CAMELS=args.CAMELS,
        CODE=args.CODE,
    )

    cosmo = train_dict["cosmo"]
    scales = train_dict["scales"]

    dm_poss = train_dict["dm_poss"][i_ref]
    dm_vels = train_dict["dm_vels"][i_ref]
    gas_poss = train_dict["gas_poss"][i_ref]
    gas_vels = train_dict["gas_vels"][i_ref]

    total_mass = cosmo.Omega_b + cosmo.Omega_c
    gas_mass = cosmo.Omega_b / total_mass

    rhos_gas = vcic_paint(jnp.zeros(mesh_shape), gas_poss, gas_mass)
    deltas_gas = rhos_gas / rhos_gas.mean() - 1

    _, cls_gas = objectives.vpower_spectrum(deltas_gas)

    train_step_fn = training.get_train_step(mesh_per_dim, cosmo)

    ref_scales = scales[i_ref]
    ref_stop = float(scales[i_ref[-1]])

    return TrainingData(
        mesh_shape=mesh_shape,
        mesh_per_dim=mesh_per_dim,
        parts_per_dim=parts_per_dim,
        i0=args.i0,
        i_ref=i_ref,
        scales=scales,
        ref_scales=ref_scales,
        ref_stop=ref_stop,
        dm_poss=dm_poss,
        dm_vels=dm_vels,
        gas_poss=gas_poss,
        gas_vels=gas_vels,
        cls_gas=cls_gas,
        deltas_gas=deltas_gas,
        train_step=train_step_fn,
    )


def init_wandb_run(
    args: argparse.Namespace,
    config_defaults: Mapping[str, Any],
    wandb_module: Optional[Any],
) -> tuple[Optional[Any], Mapping[str, Any]]:
    """Initialise a wandb run when logging is enabled and return the run and config."""
    if wandb_module is None:
        return None, config_defaults

    init_kwargs: Dict[str, Any] = {
        "config": dict(config_defaults),
        "project": "JaxHPM",
        "mode": "online",
    }
    if args.wandb_tags:
        init_kwargs["tags"] = args.wandb_tags

    wandb_run = wandb_module.init(**init_kwargs)
    wandb_module.define_metric("train/step")
    wandb_module.define_metric("train/loss", step_metric="train/step")
    wandb_module.define_metric("train/grad_norm", step_metric="train/step")

    return wandb_run, wandb_run.config


def create_pressure_components(
    args: argparse.Namespace,
    data: TrainingData,
    hparams: HyperParams,
):
    """Instantiate the pressure model, its loss function, and optimiser."""
    pressure_model = ConditionedCNN(
        d_in=4 + args.with_latent,
        d_out=1 + args.with_latent,
        d_hidden=hparams.d_hidden,
        n_hidden=hparams.n_hidden,
        kernel_size=(hparams.kernel_size,) * 3,
        rngs=nnx.Rngs(0),
        norm_type="layer",
        activation=hparams.activation_fn,
        use_residual=True,
    )

    pressure_loss_fn = ParticleLoss(
        data.mesh_per_dim,
        w_pos=1.0,
        w_vel=0.01,
        w_cls=0.1,
        w_cross=0.0,
        w_snapshot=0.0,
        k_max=2,
        loss_type="huber",
        robust_scale=data.mesh_per_dim // 16,
        cutoff_quantile=0.95,
    )

    clip_norm = 1.0
    pressure_optimizer = nnx.ModelAndOptimizer(
        pressure_model,
        optax.chain(optax.clip_by_global_norm(clip_norm), optax.adam(hparams.learning_rate)),
    )

    return pressure_model, pressure_loss_fn, pressure_optimizer


def build_initial_state(args: argparse.Namespace, data: TrainingData, pressure_model) -> tuple[Any, ...]:
    """Construct the initial state tuple passed to the ODE solver."""
    y0 = (data.dm_poss[args.i0], data.dm_vels[args.i0], data.gas_poss[args.i0], data.gas_vels[args.i0])
    if args.with_latent:
        if isinstance(pressure_model, jaxpm.nn.MLP):
            y0 += (jnp.ones((data.parts_per_dim**3, 1)),)
        elif isinstance(pressure_model, jaxpm.nn.ConditionedCNN):
            y0 += (jnp.ones(data.mesh_shape + (1,)),)
        else:
            raise NotImplementedError(
                "Latent variables only supported for jaxpm.nn.MLP and jaxpm.nn.ConditionedCNN models."
            )
    return y0


def compute_time_window(scales: jnp.ndarray, t0: float, ref_stop: float) -> jnp.ndarray:
    """Return the subset of time samples used for integration given start and stop times."""
    mask = (t0 <= scales) & (scales <= ref_stop)
    return scales[mask]


def train_once(
    args: argparse.Namespace,
    data: TrainingData,
    base_hparams: Mapping[str, Any],
    wandb_module: Optional[Any] = None,
) -> None:
    """Execute one training run, optionally reporting metrics to wandb."""
    wandb_run, config_source = init_wandb_run(args, base_hparams, wandb_module)
    hparams = HyperParams.from_config(config_source, base_hparams)

    if wandb_run is not None:
        wandb_run.config.update(hparams.as_dict(), allow_val_change=True)

    pressure_model, pressure_loss_fn, pressure_optimizer = create_pressure_components(args, data, hparams)

    train_step_fn = data.train_step

    def single_train_step(aug_key) -> tuple[float, float]:
        y0 = build_initial_state(args, data, pressure_model)
        t0 = float(data.scales[args.i0])
        tstep = compute_time_window(data.scales, t0, data.ref_stop)

        loss, grad = train_step_fn(
            pressure_loss_fn,
            pressure_optimizer,
            y0,
            t0,
            ref_t=data.ref_scales,
            ref_poss=data.gas_poss,
            ref_vels=data.gas_vels,
            ref_cls=data.cls_gas,
            ref_deltas=data.deltas_gas,
            pressure_model=pressure_model,
            gravity_model=None,
            model_to_train="pressure",
            aug_key=aug_key,
            tstep=tstep,
            nt=2,
        )

        return float(loss), float(grad)

    key = jax.random.key(71)
    progress = tqdm.tqdm(range(args.n_steps), desc="training", dynamic_ncols=True)

    for step in progress:
        key, subkey = jax.random.split(key)
        loss_value, grad_value = single_train_step(subkey)

        progress.set_postfix({"loss": f"{loss_value:.4e}", "grad": f"{grad_value:.4e}"})

        if wandb_module is not None:
            wandb_module.log(
                {
                    "train/step": step,
                    "train/loss": loss_value,
                    "train/grad_norm": grad_value,
                }
            )

    if wandb_run is not None:
        wandb_run.finish()


def import_wandb() -> Any:
    """Import wandb on demand, providing a helpful error message when unavailable."""
    try:
        import wandb  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("wandb must be installed to use --use_wandb. Install with `pip install wandb`.") from exc

    return wandb


def run_with_optional_wandb(
    args: argparse.Namespace,
    data: TrainingData,
    base_hparams: Mapping[str, Any],
    sweep_config: Optional[Dict[str, Any]],
) -> None:
    """Handle plain runs, wandb-logged runs, and wandb sweeps with a unified interface."""
    if not args.use_wandb:
        train_once(args, data, base_hparams, None)
        return

    wandb_module = import_wandb()
    running_inside_agent = "WANDB_SWEEP_ID" in os.environ

    if sweep_config is not None and not running_inside_agent:
        sweep_id = wandb_module.sweep(sweep_config, project="JaxHPM")
        print(f"Initialized wandb sweep: {sweep_id}")

        wandb_module.agent(
            sweep_id,
            function=lambda: train_once(args, data, base_hparams, wandb_module),
            count=args.sweep_count,
        )
    else:
        train_once(args, data, base_hparams, wandb_module)


def main() -> None:
    """Program entrypoint coordinating configuration, data prep, and training."""
    args = parse_args()

    if args.sweep_config and not args.use_wandb:
        raise ValueError("--sweep-config requires --use_wandb")

    base_hparams, sweep_config = load_configurations(args)
    prepare_output_directory(args.BASE, args.OUT)
    training_data = build_training_data(args)

    run_with_optional_wandb(args, training_data, base_hparams, sweep_config)


if __name__ == "__main__":
    main()
