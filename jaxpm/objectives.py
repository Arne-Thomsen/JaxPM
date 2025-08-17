import numpy as np

import jax
import jax.numpy as jnp
from typing import Optional

from jaxpm.painting import cic_paint, cic_read, compensate_cic
from jaxpm.utils import power_spectrum, cross_correlation_coefficients

box_size = 25.0  # Mpc/h

# vectorize over axis 0 of snapshots
vcic_paint = jax.vmap(cic_paint, in_axes=(None, 0, None))
vcic_read = jax.vmap(cic_read, in_axes=(0, 0))

vpower_spectrum = jax.vmap(
    lambda fields: power_spectrum(
        compensate_cic(fields),
        boxsize=np.array([box_size] * 3),
        kmin=np.pi / box_size,
        dk=2 * np.pi / box_size,
    )
)

vcross_correlation = jax.vmap(
    lambda field_a, field_b: cross_correlation_coefficients(
        compensate_cic(field_a),
        compensate_cic(field_b),
        boxsize=np.array([box_size] * 3),
        kmin=np.pi / box_size,
        dk=2 * np.pi / box_size,
    )
)


def two_point_loss(
    mesh_per_dim,
    res_poss,
    ref_cls,
    ref_deltas,
    w_cls=1.0,
    w_cross=0.0,
    w_snapshot=0.0,
    eps=1e-8,
    weight_k=True,
    debug=False,
):
    loss = 0.0

    res_rhos = vcic_paint(jnp.zeros([mesh_per_dim] * 3), res_poss, 1)
    res_deltas = res_rhos / res_rhos.mean() - 1

    # power spectrum
    if w_cls > 0.0:
        print(f"w_cls = {w_cls}")

        kbins, res_cls = vpower_spectrum(res_deltas)
        cls_loss = (res_cls / jnp.maximum(ref_cls, eps) - 1) ** 2

        if weight_k:
            k = kbins[0]
            k_min, k_cutoff = k[0], k[int(0.4 * len(k))]
            k_weights = jnp.expand_dims(jnp.exp(-((k - k_min) ** 2) / k_cutoff), 0)
            cls_loss *= k_weights

        cls_loss = jnp.sum(cls_loss, axis=-1)
        if w_snapshot > 0.0:
            cls_loss *= w_snapshot
        if not debug:
            cls_loss = jnp.mean(cls_loss)

        if debug:
            print(f"cls_loss = {w_cls * cls_loss}")

        loss += w_cls * cls_loss

        # cross correlation
        if w_cross > 0.0:
            print(f"w_cross = {w_cross}")

            kbins, res_cross = vcross_correlation(res_deltas, ref_deltas)

            cross_loss = (res_cross / jnp.sqrt(ref_cls * res_cls) - 1) ** 2
            if weight_k:
                cross_loss *= k_weights
            cross_loss = jnp.sum(cross_loss, axis=-1)
            if w_snapshot > 0.0:
                cross_loss *= w_snapshot
            if not debug:
                cross_loss = jnp.mean(cross_loss)

            if debug:
                print(f"cross_loss = {w_cross * cross_loss}")

            loss += w_cross * cross_loss
    elif w_cross > 0.0:
        raise ValueError("Cross-correlation loss is not supported without power spectrum loss.")

    return loss


class ParticleLoss:
    """Wrapper class around the functional :func:`particle_loss`.

    Designed to emulate the style of TensorFlow/Keras loss classes while reusing
    the existing implementation. Instantiate with hyper-parameters and call
    the instance to compute the loss.
    """

    def __init__(
        self,
        mesh_per_dim: int,
        w_pos: float = 1.0,
        w_vel: float = 0.0,
        w_cls: float = 0.0,
        w_cross: float = 0.0,
        w_snapshot: float = 0.0,
        cutoff_quantile: float = 0.95,
        weight_k: bool = True,
        eps: float = 1e-8,
    ) -> None:
        if w_cross > 0.0 and w_cls == 0.0:
            raise ValueError("Cross-correlation loss requires w_cls > 0.0")
        self.mesh_per_dim = mesh_per_dim
        self.w_pos = w_pos
        self.w_vel = w_vel
        self.w_cls = w_cls
        self.w_cross = w_cross
        self.w_snapshot = w_snapshot
        self.cutoff_quantile = cutoff_quantile
        self.weight_k = weight_k
        self.eps = eps

    def __call__(
        self,
        res_poss: jnp.ndarray,
        res_vels: jnp.ndarray,
        ref_poss: Optional[jnp.ndarray] = None,
        ref_vels: Optional[jnp.ndarray] = None,
        ref_cls: Optional[jnp.ndarray] = None,
        ref_deltas: Optional[jnp.ndarray] = None,
        snapshot_mean=True,
        particle_mean=True,
        debug: bool = False,
    ):

        print("using particle loss")

        loss = 0.0

        # position
        if self.w_pos > 0.0:
            print(f"w_pos = {self.w_pos}")

            dist = ((res_poss - ref_poss + self.mesh_per_dim // 2) % self.mesh_per_dim) - self.mesh_per_dim // 2
            pos_loss = jnp.sum(dist**2, axis=-1)
            pos_loss = jnp.where(pos_loss < jnp.quantile(pos_loss, self.cutoff_quantile), pos_loss, 0.0)
            if self.w_snapshot > 0.0:
                pos_loss *= self.w_snapshot

            if particle_mean:
                pos_loss = jnp.mean(pos_loss, axis=-1)
            if snapshot_mean:
                pos_loss = jnp.mean(pos_loss, axis=0)

            # pos_loss = jnp.mean(pos_loss)

            if debug:
                print(f"pos_loss = {self.w_pos * pos_loss}")

            loss += self.w_pos * pos_loss

        # velocity
        if self.w_vel > 0.0:
            print(f"w_vel = {self.w_vel}")

            vel_loss = jnp.sum((res_vels - ref_vels) ** 2, axis=-1)
            vel_loss = jnp.where(vel_loss < jnp.quantile(vel_loss, self.cutoff_quantile), vel_loss, 0.0)
            if self.w_snapshot > 0.0:
                vel_loss *= self.w_snapshot

            if particle_mean:
                vel_loss = jnp.mean(vel_loss, axis=-1)
            if snapshot_mean:
                vel_loss = jnp.mean(vel_loss, axis=0)

            if debug:
                print(f"vel_loss = {self.w_vel * vel_loss}")

            loss += self.w_vel * vel_loss

        # two-point
        if self.w_cls > 0.0 or self.w_cross > 0.0:
            loss += two_point_loss(
                self.mesh_per_dim,
                res_poss,
                ref_cls,
                ref_deltas,
                self.w_cls,
                self.w_cross,
                self.w_snapshot,
                self.eps,
                self.weight_k,
                debug,
            )

        return loss


class FieldLoss:
    """Wrapper class around the functional :func:`field_loss`.

    Designed to emulate the style of TensorFlow/Keras loss classes while reusing
    the existing implementation. Instantiate with hyper-parameters and call
    the instance to compute the loss.
    """

    def __init__(
        self,
        mesh_per_dim: int,
        w_field: float = 1.0,
        w_vel: float = 0.0,
        w_cls: float = 0.0,
        w_cross: float = 0.0,
        w_snapshot: float = 0.0,
        weight_k: bool = True,
        eps: float = 1e-8,
    ) -> None:
        if w_cross > 0.0 and w_cls == 0.0:
            raise ValueError("Cross-correlation loss requires w_cls > 0.0")
        self.mesh_per_dim = mesh_per_dim
        self.w_field = w_field
        self.w_vel = w_vel
        self.w_cls = w_cls
        self.w_cross = w_cross
        self.w_snapshot = w_snapshot
        self.eps = eps

    def __call__(
        self,
        res_poss: jnp.ndarray,
        res_vels: jnp.ndarray,
        ref_poss: Optional[jnp.ndarray] = None,
        ref_vels: Optional[jnp.ndarray] = None,
        ref_cls: Optional[jnp.ndarray] = None,
        ref_deltas: Optional[jnp.ndarray] = None,
        debug: bool = False,
    ):
        print("using field loss")

        loss = 0.0

        if self.w_field > 0.0:
            print(f"w_field = {self.w_field}")

            res_rhos = vcic_paint(jnp.zeros([self.mesh_per_dim] * 3), res_poss, 1)
            res_deltas = res_rhos / res_rhos.mean() - 1

            field_loss = (res_deltas - ref_deltas) ** 2
            field_loss = jnp.where(field_loss < jnp.quantile(field_loss, 0.95), field_loss, 0.0)
            # rho_loss /= jnp.maximum(scales.reshape(-1,1,1,1)**2, eps)
            field_loss = jnp.mean(field_loss)

            if debug:
                print(f"field_loss = {self.w_field * field_loss}")

            loss += self.w_field * field_loss

        # two-point
        if self.w_cls > 0.0 or self.w_cross > 0.0:
            loss += two_point_loss(
                self.mesh_per_dim,
                res_poss,
                ref_cls,
                ref_deltas,
                self.w_cls,
                self.w_cross,
                self.w_snapshot,
                self.eps,
                weight_k,
                debug,
            )

        return loss
