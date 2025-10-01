import jax
import jax.numpy as jnp

from diffrax import diffeqsolve, ODETerm, LeapfrogMidpoint, SaveAt, ConstantStepSize, StepTo
from flax import nnx
import orbax.checkpoint as ocp

from jaxpm import hpm, utils


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


# TODO integrate into network classes
def load_checkpoint(model, checkpoint_file):
    abstract_model = nnx.eval_shape(lambda: model)
    graphdef, abstract_params = nnx.split(abstract_model)

    checkpointer = ocp.StandardCheckpointer()
    params = checkpointer.restore(checkpoint_file, abstract_params)
    model = nnx.merge(graphdef, params)
    print(f"Checkpoint loaded from {checkpoint_file}")

    return model


def save_checkpoint(model, checkpoint_file):
    _, params = nnx.split(model)
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(checkpoint_file, params, force=True)
    print(f"Checkpoint saved to {checkpoint_file}")
