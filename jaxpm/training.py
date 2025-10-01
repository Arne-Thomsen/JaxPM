import jax
import jax.numpy as jnp

from diffrax import diffeqsolve, ODETerm, LeapfrogMidpoint, SaveAt, ConstantStepSize, StepTo
from flax import nnx

from jaxpm import hpm, utils, augmentations


def get_ode_solver(mesh_per_dim, cosmo, max_steps=1000):

    def solve_ode(y0, t0, tsave, gravity_model, pressure_model, training=True, dt0=None, nt=2, tstep=None):
        ode = ODETerm(
            hpm.get_hpm_network_ode_fn(
                mesh_per_dim,
                cosmo,
                gravity_model=gravity_model,
                pressure_model=pressure_model,
                training=training,
            )
        )

        if dt0 is not None:
            t1 = tsave[-1]
            stepsize_controller = ConstantStepSize()
            dt0 = dt0
            print(f"Solving ODE in fixed steps (ConstantStepSize with dt0={dt0})")

        elif nt is not None:
            if tstep is None:
                tstep = tsave
            tstep = utils.refine_time_steps(tstep, nt)
            t1 = tstep[-1]
            stepsize_controller = StepTo(tstep)
            dt0 = None
            print(f"Solving ODE in {len(tstep)} steps (StepTo with {nt} steps between ts)")

        res = diffeqsolve(
            terms=ode,
            y0=y0,
            t0=t0,
            t1=t1,
            solver=LeapfrogMidpoint(),
            saveat=SaveAt(ts=tsave),
            max_steps=max_steps,
            dt0=dt0,
            stepsize_controller=stepsize_controller,
        )
        res = res.ys

        res = jnp.stack([jnp.squeeze(r) for r in res])

        return res

    return solve_ode


def train_step_base(model, optimizer, loss_fn):
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)

    squared_sum = jax.tree_util.tree_reduce(lambda x, y: x + jnp.sum(y**2), grads, 0.0)
    grad_norm = jnp.sqrt(squared_sum)

    return loss, grad_norm


def get_train_step(mesh_per_dim, cosmo, max_steps=1000):
    solve_ode = get_ode_solver(mesh_per_dim, cosmo, max_steps=max_steps)

    def model_loss_base_fn(
        loss_fn,
        y0,
        t0,
        # reference
        ref_t,
        ref_poss,
        ref_vels=None,
        ref_cls=None,
        ref_deltas=None,
        # models
        gravity_model=None,
        pressure_model=None,
        species="gas",
        # ode
        tstep=None,
        nt=2,
        aug_key=None,
    ):
        if aug_key is not None:
            print("Applying augmentations (random flips and 90 degree rotations)")

            # initial conditions
            dm_aug = augmentations.rot_flip_3d(aug_key, mesh_per_dim, pos=y0[0], vel=y0[1])
            if len(y0) == 2:
                y0 = (dm_aug["pos"], dm_aug["vel"])
            elif len(y0) == 4:
                gas_aug = augmentations.rot_flip_3d(aug_key, mesh_per_dim, pos=y0[2], vel=y0[3])
                y0 = (dm_aug["pos"], dm_aug["vel"], gas_aug["pos"], gas_aug["vel"])
            else:
                raise NotImplementedError("Augmentations for latent variables not implemented yet")

            # reference
            ref_poss = augmentations.rot_flip_3d(aug_key, mesh_per_dim, pos=ref_poss)["pos"]
            if ref_vels is not None:
                ref_vels = augmentations.rot_flip_3d(aug_key, mesh_per_dim, vel=ref_vels)["vel"]
            if ref_deltas is not None:
                ref_deltas = augmentations.rot_flip_3d(aug_key, mesh_per_dim, field=ref_deltas)["field"]

        # integrate
        res = solve_ode(y0, t0, ref_t, gravity_model, pressure_model, tstep=tstep, nt=nt)
        if species == "dm":
            res_poss, res_vels = res[0], res[1]
        elif species == "gas":
            res_poss, res_vels = res[2], res[3]
        else:
            raise NotImplementedError

        return loss_fn(
            res_poss=res_poss,
            res_vels=res_vels,
            ref_poss=ref_poss,
            ref_vels=ref_vels,
            ref_cls=ref_cls,
            ref_deltas=ref_deltas,
        )

    @nnx.jit(static_argnames=("loss_fn", "model_to_train", "nt"))
    def train_step(
        loss_fn,
        optimizer,
        y0,
        t0,
        ref_t,
        ref_poss,
        ref_vels=None,
        ref_cls=None,
        ref_deltas=None,
        gravity_model=None,
        pressure_model=None,
        model_to_train="pressure",
        tstep=None,
        nt=2,
        aug_key=None,
    ):
        """dynamic snapshot range as passed"""

        if model_to_train == "pressure":

            def model_loss_fn(model):
                return model_loss_base_fn(
                    loss_fn,
                    y0,
                    t0,
                    ref_t,
                    ref_poss,
                    ref_vels,
                    ref_cls,
                    ref_deltas,
                    gravity_model=gravity_model,
                    pressure_model=model,
                    species="gas",
                    tstep=tstep,
                    nt=nt,
                    aug_key=aug_key,
                )

            return train_step_base(pressure_model, optimizer, model_loss_fn)

        elif model_to_train == "gravity":

            def model_loss_fn(model):
                return model_loss_base_fn(
                    loss_fn,
                    y0,
                    t0,
                    ref_t,
                    ref_poss,
                    ref_vels,
                    ref_cls,
                    ref_deltas,
                    gravity_model=model,
                    pressure_model=pressure_model,
                    species="dm",
                    tstep=tstep,
                    nt=nt,
                    aug_key=aug_key,
                )

            return train_step_base(gravity_model, optimizer, model_loss_fn)

    return train_step
