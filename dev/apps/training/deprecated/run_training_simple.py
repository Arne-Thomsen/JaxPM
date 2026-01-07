import os, tqdm, argparse

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


def setup():
    parser = argparse.ArgumentParser()

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
    parser.add_argument(
        "--parts_per_dim",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--with_latent",
        action="store_true",
    )
    parser.add_argument(
        "--i0",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--n_ref",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=100,
    )

    args, _ = parser.parse_known_args()

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

    OUT = os.path.join(args.BASE, args.OUT)
    os.makedirs(OUT, exist_ok=True)
    print(f"Output directory: {OUT}")

    parts_per_dim = args.parts_per_dim
    mesh_per_dim = parts_per_dim
    mesh_shape = [mesh_per_dim] * 3

    i0 = args.i0
    i_ref = np.linspace(0, 33, args.n_ref + 1, dtype=int)[1:]

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

    pressure_model = ConditionedCNN(
        d_in=4 + args.with_latent,
        d_out=1 + args.with_latent,
        d_hidden=64,
        n_hidden=4,
        kernel_size=(3, 3, 3),
        rngs=nnx.Rngs(0),
        norm_type="layer",
        activation=jax.nn.swish,
        use_residual=True,
    )

    pressure_loss_fn = ParticleLoss(
        mesh_per_dim,
        w_pos=1.0,
        w_vel=0.01,
        w_cls=0.1,
        w_cross=0.0,
        w_snapshot=0.0,
        k_max=2,
        loss_type="huber",
        robust_scale=mesh_per_dim // 16,
        cutoff_quantile=0.95,
    )

    learning_rate = 1e-4
    clip_norm = 1
    pressure_optimizer = nnx.ModelAndOptimizer(
        pressure_model, optax.chain(optax.clip_by_global_norm(clip_norm), optax.adam(learning_rate))
    )

    def train_step_wrapper(i0, aug_key=None):
        y0 = (dm_poss[i0], dm_vels[i0], gas_poss[i0], gas_vels[i0])
        if args.with_latent:
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

        pbar.set_description(f"[{i_ref[0]}, {i_ref[-1]}], Loss: {loss:.4e}, Grad: {grad:.4e}")

    key = jax.random.key(71)
    for i in (pbar := tqdm.tqdm(range(args.n_steps))):
        key, subkey = jax.random.split(key)
        train_step_wrapper(i0=i0, aug_key=subkey)
