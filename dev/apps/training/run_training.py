"""Training entry point for pressure model fine-tuning on CAMELS snapshots.

Usage with fixed hyperparameters from YAML configuration:
python run_training.py \
    --hparams_config configs/debug/hparams_cnn.yaml \
    --sim_config configs/sim.yaml \
    --loss_config configs/loss.yaml \
    --use_wandb

python run_training.py \
    --hparams_config configs/hparams_cnn_128.yaml \
    --sim_config configs/sim_128.yaml \
    --loss_config configs/particle_loss.yaml \
    --use_wandb

Usage with wandb sweep (sweep name defined in hparams_sweep.yaml):
python run_training.py \
    --hparams_sweep_config configs/hparams_sweep.yaml \
    --sim_config configs/sim.yaml \
    --loss_config configs/loss.yaml \
    --use_wandb

python run_training.py \
    --hparams_sweep_config configs/debug/hparams_sweep.yaml \
    --sim_config configs/sim.yaml \
    --loss_config configs/loss.yaml \
    --use_wandb
"""

import os, argparse, tqdm, yaml
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

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


def load_yaml_config(path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    file_path = Path(path).expanduser()
    if not file_path.exists():
        raise FileNotFoundError(f"YAML config not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:

        print(f"Loaded config from {path}")
        return yaml.safe_load(f) or {}


def get_config_value(config: Any, key: str, default: Any = None) -> Any:
    """Get value from config object (dict, SimpleNamespace, or wandb.config)."""
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def setup():
    parser = argparse.ArgumentParser()

    # paths
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
        "--hparams_config",
        type=str,
        default=None,
        help="Path to a YAML file defining hyperparameters (n_steps, d_hidden, n_hidden, kernel_size, activation, learning_rate). Required unless --sweep_net_config is provided.",
    )
    parser.add_argument(
        "--hparams_sweep_config",
        type=str,
        default=None,
        help="Path to a YAML file describing a wandb sweep for network hyperparameters (parameter ranges, search strategy, etc). When provided, --net_config is not required.",
    )
    parser.add_argument(
        "--sim_config",
        type=str,
        required=True,
        help="Path to a YAML file defining simulation parameters (parts_per_dim, with_latent, i0).",
    )
    parser.add_argument(
        "--loss_config",
        type=str,
        required=True,
        help="Path to a YAML file defining ParticleLoss parameters (weights, k_max, loss options, i_ref).",
    )
    # wandb
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging and allow sweeps to control hyperparameters.",
    )
    parser.add_argument(
        "--sweep_id",
        type=str,
        default=None,
        help="Existing wandb sweep ID to connect to (for parallel sweep agents). When provided, this agent will connect to an existing sweep instead of creating a new one.",
    )
    parser.add_argument(
        "--sweep_name",
        type=str,
        default=None,
        help="Human-readable name for the sweep. Used for organizing checkpoints and outputs. If not provided, uses the sweep ID or 'no_sweep'.",
    )
    parser.add_argument(
        "--sweep_count",
        type=int,
        default=None,
        help="Optional maximum number of sweep runs to execute when --sweep_net_config is provided.",
    )
    parser.add_argument(
        "--wandb_tags",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of tags to attach to the wandb run.",
    )

    args, _ = parser.parse_known_args()

    if not args.hparams_config and not args.hparams_sweep_config and not args.sweep_id:
        raise ValueError("Either --hparams_config, --sweep_hparams_config, or --sweep_id must be provided")

    if args.hparams_config and args.hparams_sweep_config:
        raise ValueError("Cannot provide both --hparams_config and --sweep_hparams_config; use only one")

    if args.hparams_sweep_config and not args.use_wandb:
        raise ValueError("--sweep_net_config requires --use_wandb")

    if args.sweep_id and not args.use_wandb:
        raise ValueError("--sweep_id requires --use_wandb")

    return args


def get_data(args, mesh_per_dim: int, parts_per_dim: int):
    """Load training and validation data from CAMELS."""
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


def prepare_reference_data(train_dict: Dict, i_ref: np.ndarray, mesh_shape: list):
    """Extract and prepare training data (particles and fields)."""

    cosmo = train_dict["cosmo"]
    scales = train_dict["scales"][i_ref]

    # Extract particle positions and velocities
    dm_poss = train_dict["dm_poss"][i_ref]
    dm_vels = train_dict["dm_vels"][i_ref]
    gas_poss = train_dict["gas_poss"][i_ref]
    gas_vels = train_dict["gas_vels"][i_ref]

    # Compute density fields
    dm_mass = cosmo.Omega_c / (cosmo.Omega_b + cosmo.Omega_c)
    gas_mass = 1 - dm_mass

    rhos_dm = vcic_paint(jnp.zeros(mesh_shape), dm_poss, dm_mass)
    deltas_dm = rhos_dm / rhos_dm.mean() - 1

    rhos_gas = vcic_paint(jnp.zeros(mesh_shape), gas_poss, gas_mass)
    deltas_gas = rhos_gas / rhos_gas.mean() - 1

    # Compute power spectra
    _, cls_dm = objectives.vpower_spectrum(deltas_dm)
    _, cls_gas = objectives.vpower_spectrum(deltas_gas)

    return {
        "scales": scales,
        "dm_poss": dm_poss,
        "dm_vels": dm_vels,
        "gas_poss": gas_poss,
        "gas_vels": gas_vels,
        "deltas_gas": deltas_gas,
        "cls_gas": cls_gas,
    }


def create_loss_function(loss_config: Dict[str, Any], mesh_per_dim: int) -> ParticleLoss:
    """Create the particle loss function from config."""

    if loss_config["type"] == "particle":
        loss_fn = ParticleLoss(
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

    elif loss_config["type"] == "field":
        loss_fn = FieldLoss(
            mesh_per_dim,
            w_field=float(loss_config["w_field"]),
            w_cls=float(loss_config["w_cls"]),
            w_cross=float(loss_config["w_cross"]),
            w_snapshot=float(loss_config["w_snapshot"]),
            k_max=int(loss_config["k_max"]),
            loss_type=str(loss_config["loss_type"]),
            robust_scale=float(loss_config["robust_scale"]),
            cutoff_quantile=float(loss_config["cutoff_quantile"]),
        )

    else:
        raise ValueError(f"Unknown loss type: {loss_config['type']}")

    return loss_fn


def create_pressure_model(hparams: Dict[str, Any]) -> ConditionedCNN:
    """Create and initialize the pressure model."""

    with_latent = bool(hparams["with_latent"])
    architecture = hparams["architecture"]

    # NOTE hardcoded feature dimensions
    if architecture == "cnn":
        model = ConditionedCNN(
            d_in=4 + int(with_latent),
            d_out=1 + int(with_latent),
            d_hidden=hparams["d_hidden"],
            n_hidden=hparams["n_hidden"],
            kernel_size=(hparams["kernel_size"],) * 3,
            activation=ACTIVATIONS[hparams["activation"]],
            use_residual=True,
            norm_type="layer",
            rngs=nnx.Rngs(0),
        )
    elif architecture == "mlp":
        model = MLP(
            d_in=5 + int(with_latent),
            d_out=1 + int(with_latent),
            d_hidden=hparams["d_hidden"],
            n_hidden=hparams["n_hidden"],
            activation=ACTIVATIONS[hparams["activation"]],
            norm_type="layer",
            rngs=nnx.Rngs(0),
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    return model


def create_optimizer(hparams: Dict[str, Any]) -> optax.GradientTransformation:
    """Create the optimizer."""

    learning_rate = hparams["learning_rate"]

    if hparams["cosine_decay"]:
        n_steps = hparams["n_steps"]
        learning_rate = optax.cosine_decay_schedule(
            init_value=learning_rate,
            decay_steps=n_steps,
            alpha=0.0,
        )

    optimizer = hparams["optimizer"]
    if hparams["optimizer"] == "adam":
        optimizer = optax.adam
    elif hparams["optimizer"] == "adamw":
        optimizer = optax.adamw

    return optax.chain(optax.clip_by_global_norm(1.0), optimizer(learning_rate))


def initialize_wandb(
    wandb_module: Any,
    config_defaults: Dict[str, Any],
    sim_config: Dict[str, Any],
    loss_config: Dict[str, Any],
    tags: Optional[list] = None,
):
    """Initialize wandb run with config and metrics."""

    full_config = {
        **config_defaults,
        "sim": sim_config,
        "loss": loss_config,
    }

    wandb_init_kwargs = {
        "config": full_config,
        "project": "JaxHPM",
        "mode": "online",
    }
    if tags is not None:
        wandb_init_kwargs["tags"] = tags

    wandb_run = wandb_module.init(**wandb_init_kwargs)

    wandb_module.define_metric("train/step")
    wandb_module.define_metric("train/loss", step_metric="train/step")
    wandb_module.define_metric("train/grad_norm", step_metric="train/step")

    return wandb_run


if __name__ == "__main__":
    args = setup()

    # Load configs
    hparams_config = None
    if args.hparams_config:
        hparams_config = load_yaml_config(args.hparams_config)

    hparams_sweep_config = None
    if args.hparams_sweep_config:
        hparams_sweep_config = load_yaml_config(args.hparams_sweep_config)

    sim_config = load_yaml_config(args.sim_config)
    loss_config = load_yaml_config(args.loss_config)

    # Read configs
    parts_per_dim = int(sim_config["parts_per_dim"])
    mesh_per_dim = int(sim_config["mesh_per_dim"])
    mesh_shape = [mesh_per_dim] * 3
    i0 = int(sim_config["i0"])
    nt = int(sim_config["nt"])
    i_plot = sim_config["i_plot"]

    i_ref = np.array(loss_config["i_ref"], dtype=int)

    # Setup constants
    train_dict, vali_dict = get_data(args, mesh_per_dim, parts_per_dim)
    ref_dict = prepare_reference_data(train_dict, i_ref, mesh_shape)
    cosmo = train_dict["cosmo"]
    scales = train_dict["scales"]

    solve_ode = training.get_ode_solver(mesh_per_dim, cosmo)
    train_step = training.get_train_step(mesh_per_dim, cosmo)

    def run_training_instance(wandb_module: Optional[Any] = None):
        """Run a single training instance, optionally with wandb logging."""

        running_inside_sweep = "WANDB_SWEEP_ID" in os.environ

        # Initialize wandb or use defaults
        if wandb_module is not None:
            if running_inside_sweep:
                wandb_run = wandb_module.init()
                # Define metrics for sweep runs
                wandb_module.define_metric("train/step")
                wandb_module.define_metric("train/loss", step_metric="train/step")
                wandb_module.define_metric("train/grad_norm", step_metric="train/step")
                # Log sim and loss configs even in sweep mode
                wandb_run.config.update({"sim": sim_config, "loss": loss_config}, allow_val_change=False)
                config_source = wandb_run.config
            else:
                if hparams_config is None:
                    raise ValueError("--hparams_config is required when not running in sweep mode")
                wandb_run = initialize_wandb(wandb_module, hparams_config, sim_config, loss_config, args.wandb_tags)
                config_source = wandb_run.config
        else:
            if hparams_config is None:
                raise ValueError("--hparams_config is required when not using wandb")
            wandb_run = None
            config_source = SimpleNamespace(**hparams_config)

        # Get hyperparameters directly from config_source
        # Convert to dict for easier access
        if isinstance(config_source, SimpleNamespace):
            hparams = vars(config_source)
        elif hasattr(config_source, "as_dict"):
            hparams = config_source.as_dict()
        elif hasattr(config_source, "to_dict"):
            hparams = config_source.to_dict()
        else:
            # For wandb.config, just use it directly as a dict-like object
            hparams = {
                k: get_config_value(config_source, k)
                for k in [
                    "activation",
                    "architecture",
                    "cosine_decay",
                    "d_hidden",
                    "n_hidden",
                    "kernel_size",
                    "learning_rate",
                    "n_steps",
                    "with_latent",
                ]
            }

        with_latent = bool(hparams["with_latent"])

        # Only update config if not in a sweep (sweep values are locked)
        if wandb_run is not None and not running_inside_sweep:
            wandb_run.config.update(hparams, allow_val_change=True)

        # Always update summary with final hyperparameters for easy viewing
        if wandb_run is not None:
            wandb_run.summary.update({f"hyperparameters/{k}": v for k, v in hparams.items()})

        pressure_loss_fn = create_loss_function(loss_config, mesh_per_dim)
        pressure_model = create_pressure_model(hparams)
        pressure_optimizer = nnx.ModelAndOptimizer(pressure_model, create_optimizer(hparams))

        def train_step_wrapper(i0, aug_key=None):
            """Wrapper for training step with initial conditions and augmentation."""

            y0 = (
                train_dict["dm_poss"][i0],
                train_dict["dm_vels"][i0],
                train_dict["gas_poss"][i0],
                train_dict["gas_vels"][i0],
            )

            if with_latent:
                if isinstance(pressure_model, jaxpm.nn.MLP):
                    y0 += (jnp.ones((parts_per_dim**3, 1)),)
                elif isinstance(pressure_model, jaxpm.nn.ConditionedCNN):
                    y0 += (jnp.ones(mesh_shape + [1]),)
                else:
                    raise NotImplementedError

            # Define time steps for integration
            t0 = scales[i0]
            t1 = scales[i_ref[-1]]
            tstep = scales[(t0 <= scales) & (scales <= t1)]

            loss, grad = train_step(
                pressure_loss_fn,
                pressure_optimizer,
                y0,
                t0,
                ref_t=ref_dict["scales"],
                ref_poss=ref_dict["gas_poss"],
                ref_vels=ref_dict["gas_vels"],
                ref_cls=ref_dict["cls_gas"],
                ref_deltas=ref_dict["deltas_gas"],
                pressure_model=pressure_model,
                gravity_model=None,
                model_to_train="pressure",
                aug_key=aug_key,
                tstep=tstep,
                nt=nt,
            )

            return loss, grad

        # Training loop
        key = jax.random.key(71)
        for i in (pbar := tqdm.tqdm(range(hparams["n_steps"]))):
            key, subkey = jax.random.split(key)
            loss_value, grad_value = train_step_wrapper(i0=i0, aug_key=subkey)

            if jnp.isnan(grad_value) or jnp.isnan(loss_value):
                error_msg = f"NaN detected at step {i}: loss={loss_value:.4e}, grad_norm={grad_value:.4e}"
                print(f"\nTraining failed: {error_msg}")
                if wandb_module is not None:
                    wandb_module.log({"train/step": i, "train/loss": loss_value, "train/grad_norm": grad_value})
                    wandb_module.finish(exit_code=1)
                raise ValueError(error_msg)

            pbar.set_description(f"Step {i}")
            pbar.set_postfix({"loss": f"{loss_value:.4e}", "grad": f"{grad_value:.4e}"})

            if wandb_module is not None:
                wandb_module.log({"train/step": i, "train/loss": loss_value, "train/grad_norm": grad_value})

        if wandb_run is not None:
            if args.sweep_name:
                sweep_name = args.sweep_name
            elif wandb_run.sweep_id:
                sweep_name = wandb_run.sweep_id
            else:
                sweep_name = "no_sweep"
            run_name = wandb_run.name or wandb_run.id
        else:
            sweep_name = args.sweep_name or "no_sweep"
            run_name = "no_wandb"

        run_dir = os.path.join(os.getcwd(), "sweeps", sweep_name, run_name)
        os.makedirs(run_dir, exist_ok=True)

        # Checkpoint
        checkpoint_file = os.path.join(run_dir, f"checkpoint.jx")
        pressure_model.save(checkpoint_file)

        # Validation
        vali_loss = diagnostics.run_simulations(
            vali_dict,
            mesh_per_dim,
            pressure_model=pressure_model,
            i_init=i0,
            i_plot=i_plot,
            nt=nt,
            plot_dm=True,
            plot_gas=True,
            with_latent=with_latent,
            plot_latent=with_latent,
            i_ref=i_ref,
            loss_fn=pressure_loss_fn,
            out_dir=run_dir,
        )

        if wandb_run is not None:
            wandb_module.log({"vali/loss": vali_loss})
            wandb_run.summary["vali/loss"] = vali_loss
            wandb_run.summary["checkpoint_path"] = checkpoint_file
            wandb_module.finish()

    # Execute training (with or without wandb)
    if args.use_wandb:
        import wandb as wandb_module

        # Check if we're running inside a sweep agent
        running_inside_sweep = "WANDB_SWEEP_ID" in os.environ

        # Determine if we should run a sweep or single training
        if args.sweep_id:
            # Connect to existing sweep (for parallel agents)
            print(f"Connecting to existing sweep: {args.sweep_id}")
            wandb_module.agent(
                args.sweep_id,
                function=lambda: run_training_instance(wandb_module),
                project="JaxHPM",
                count=args.sweep_count,
            )
        elif hparams_sweep_config is not None and not running_inside_sweep:
            # Create new sweep and run agent
            # Inject sweep name from command line if provided
            if args.sweep_name:
                sweep_config = hparams_sweep_config.copy()
                sweep_config["name"] = args.sweep_name
            else:
                sweep_config = hparams_sweep_config

            sweep_id = wandb_module.sweep(sweep_config, project="JaxHPM")
            print(f"Initialized wandb sweep: {sweep_id}")
            if args.sweep_name:
                print(f"Sweep name: {args.sweep_name}")

            wandb_module.agent(
                sweep_id,
                function=lambda: run_training_instance(wandb_module),
                project="JaxHPM",
                count=args.sweep_count,
            )
        else:
            # Run single training instance with wandb
            run_training_instance(wandb_module)
    else:
        # Run without wandb
        run_training_instance(None)
