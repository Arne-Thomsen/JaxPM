import numpy as np

import jax
import jax.numpy as jnp
from typing import Optional

from jaxpm import utils
from jaxpm.painting import cic_paint, cic_read, compensate_cic

# hardcoded for CAMELS
box_size = 25.0  # Mpc/h

# vectorize over axis 0 of snapshots
vcic_paint = jax.vmap(cic_paint, in_axes=(None, 0, None))
vcic_read = jax.vmap(cic_read, in_axes=(0, 0))

power_spectrum = lambda field: utils.power_spectrum(
    compensate_cic(field),
    boxsize=np.array([box_size] * 3),
    kmin=np.pi / box_size,
    dk=2 * np.pi / box_size,
)
vpower_spectrum = jax.vmap(power_spectrum)

cross_correlation = lambda field_a, field_b: utils.cross_correlation_coefficients(
    compensate_cic(field_a),
    compensate_cic(field_b),
    boxsize=np.array([box_size] * 3),
    kmin=np.pi / box_size,
    dk=2 * np.pi / box_size,
)
vcross_correlation = jax.vmap(cross_correlation)


def two_point_loss(
    mesh_per_dim,
    res_poss,
    ref_cls,
    ref_deltas,
    w_cls=1.0,
    w_cross=0.0,
    w_snapshot=0.0,
    k_max=None,
    k_type="step",
    eps=1e-8,
    debug=False,
):
    loss = 0.0

    vmap_snapshots = res_poss.ndim == 3
    if vmap_snapshots:
        res_rhos = vcic_paint(jnp.zeros([mesh_per_dim] * 3), res_poss, 1)
    else:
        res_rhos = cic_paint(jnp.zeros([mesh_per_dim] * 3), res_poss, 1)
    res_deltas = res_rhos / res_rhos.mean() - 1

    # power spectrum
    if w_cls > 0.0:
        print(f"w_cls = {w_cls}, k_max = {k_max}, k_type = {k_type}")
        assert ref_cls is not None, "ref_cls required for power spectrum loss"

        if vmap_snapshots:
            kbins, res_cls = vpower_spectrum(res_deltas)
            k = kbins[0]
        else:
            kbins, res_cls = power_spectrum(res_deltas)
            k = kbins
        cls_loss = (res_cls / jnp.maximum(ref_cls, eps) - 1) ** 2

        if k_max is not None:
            k_min = k[0]
            if k_type == "step":
                k_weights = jnp.where(k < k_max, 1.0, 0.0)
            elif k_type == "gaussian":
                k_weights = jnp.exp(-((k - k_min) ** 2) / k_max)
            elif k_type == "sigmoid":
                k_weights = 1 / (1 + jnp.exp(jnp.median(k) * (k - k_max)))
            elif k_type == "hyperbola":
                k_weights = 1 / jnp.sqrt(k)
                k_weights = jnp.where(k < k_max, k_weights, 0.0)
            else:
                raise ValueError

            cls_loss *= k_weights

        cls_loss = jnp.sum(cls_loss, axis=-1)
        if w_snapshot > 0.0:
            cls_loss *= w_snapshot
        cls_loss = jnp.mean(cls_loss)

        if debug:
            print(f"cls_loss = {(w_cls * cls_loss):.4e}")

        loss += w_cls * cls_loss

        # cross correlation
        if w_cross > 0.0:
            print(f"w_cross = {w_cross}")
            assert ref_deltas is not None, "ref_deltas required for cross-correlation loss"

            if vmap_snapshots:
                _, res_cross = vcross_correlation(res_deltas, ref_deltas)
            else:
                _, res_cross = cross_correlation(res_deltas, ref_deltas)
            cross_loss = (res_cross / jnp.sqrt(ref_cls * res_cls) - 1) ** 2

            if k_max is not None:
                cross_loss *= k_weights

            cross_loss = jnp.sum(cross_loss, axis=-1)
            if w_snapshot > 0.0:
                cross_loss *= w_snapshot
            cross_loss = jnp.mean(cross_loss)

            if debug:
                print(f"cross_loss = {(w_cross * cross_loss):.4e}")

            loss += w_cross * cross_loss

    elif w_cross > 0.0:
        raise ValueError("Cross-correlation loss is not supported without power spectrum loss.")

    return loss


class ParticleLoss:
    """Particle-based loss function for cosmological simulations.

    Computes losses based on particle positions, velocities, pressure, and
    two-point statistics. Supports robust loss types and periodic
    boundary conditions for positions.

    Args:
        mesh_per_dim: Number of mesh cells per dimension
        w_pos: Weight for position loss
        w_vel: Weight for velocity loss
        w_cls: Weight for power spectrum loss
        w_cross: Weight for cross-correlation loss
        w_snapshot: Weight for snapshot-specific losses
        w_P: Weight for pressure loss
        cutoff_quantile: Quantile cutoff for outlier suppression
        weight_k: Whether to apply k-weighting in power spectrum
        loss_type: Type of pointwise residual loss to use for positions and
            velocities. One of {'l2', 'huber', 'geman'}.
        robust_scale: Robustness scale parameter used by 'huber' (delta) and
            'geman' (c). Defaults to mesh_per_dim // 16 if not provided.
        eps: Small epsilon for numerical stability
    """

    def __init__(
        self,
        mesh_per_dim: int,
        w_pos: float = 1.0,
        w_vel: float = 0.0,
        w_cls: float = 0.0,
        w_cross: float = 0.0,
        w_snapshot: float = 0.0,
        w_P: float = 0.0,
        k_max: Optional[float] = None,
        k_type: str = "step",
        cutoff_quantile: Optional[float] = None,
        loss_type: str = "huber",
        robust_scale: Optional[float] = None,
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
        self.w_P = w_P
        self.cutoff_quantile = cutoff_quantile
        self.k_max = k_max
        self.k_type = k_type

        inferred = str(loss_type).lower()
        if inferred not in {"l2", "huber", "geman"}:
            raise ValueError(f"Unsupported loss_type '{loss_type}'. Choose from 'l2', 'huber', 'geman'.")
        self.loss_type = inferred
        default_scale = max(1, mesh_per_dim // 16)
        self.robust_scale = float(robust_scale) if robust_scale is not None else float(default_scale)
        self.eps = eps

    def _robust_loss(self, dist: jnp.ndarray, robust_scale=None) -> jnp.ndarray:
        """Pointwise robust loss for residuals.

        Applies the selected loss_type with self.robust_scale as the
        characteristic scale. Returns per-component losses, caller can reduce
        across axes as needed.
        """
        if robust_scale is None:
            robust_scale = self.robust_scale

        lt = self.loss_type
        if lt == "l2":
            return dist**2
        if lt == "huber":
            delta = robust_scale
            abs_dist = jnp.abs(dist)
            return jnp.where(abs_dist <= delta, 0.5 * dist**2, delta * (abs_dist - 0.5 * delta))
        if lt == "geman":
            # Geman–McClure: rho(r) = r^2 / (r^2 + c^2)
            c2 = robust_scale**2
            return (dist**2) / (dist**2 + c2 + self.eps)
        # Should not happen due to validation in __init__
        raise ValueError(f"Unknown loss_type: {lt}")

    def _apply_loss_reduction(self, loss: jnp.ndarray, particle_mean: bool, snapshot_mean: bool) -> jnp.ndarray:
        if particle_mean:
            loss = jnp.mean(loss, axis=-1)
        if snapshot_mean:
            loss = jnp.mean(loss, axis=0)
        return loss

    def __call__(
        self,
        res_poss: Optional[jnp.ndarray] = None,
        res_vels: Optional[jnp.ndarray] = None,
        res_Ps: Optional[jnp.ndarray] = None,
        ref_poss: Optional[jnp.ndarray] = None,
        ref_vels: Optional[jnp.ndarray] = None,
        ref_cls: Optional[jnp.ndarray] = None,
        ref_deltas: Optional[jnp.ndarray] = None,
        ref_Ps: Optional[jnp.ndarray] = None,
        snapshot_mean: bool = True,
        particle_mean: bool = True,
        debug: bool = False,
    ) -> jnp.ndarray:
        print("Using particle loss")

        loss = 0.0

        # Position loss
        if self.w_pos > 0.0:
            if res_poss is None or ref_poss is None:
                raise ValueError("Position loss requires both res_poss and ref_poss")
            print(f"w_pos = {self.w_pos}, loss = {self.loss_type}, scale = {self.robust_scale}")

            # Compute distance with periodic boundary conditions
            dist = ((res_poss - ref_poss + self.mesh_per_dim // 2) % self.mesh_per_dim) - self.mesh_per_dim // 2

            # Apply selected robust loss component-wise, then sum over spatial dims
            pos_loss = self._robust_loss(dist)
            pos_loss = jnp.sum(pos_loss, axis=-1)

            if self.cutoff_quantile is not None:
                cutoff = jnp.quantile(pos_loss, self.cutoff_quantile)
                pos_loss = jnp.where(pos_loss < cutoff, pos_loss, 0.0)

            if self.w_snapshot > 0.0:
                pos_loss *= self.w_snapshot

            pos_loss = self._apply_loss_reduction(pos_loss, particle_mean, snapshot_mean)

            if debug:
                print(f"pos_loss = {(self.w_pos * pos_loss):.4e}")

            loss += self.w_pos * pos_loss

        # Velocity loss
        if self.w_vel > 0.0:
            if res_vels is None or ref_vels is None:
                raise ValueError("Velocity loss requires both res_vels and ref_vels")
            vel_robust_scale = 4 * self.robust_scale
            print(f"w_vel = {self.w_vel}, loss = {self.loss_type}, scale = {vel_robust_scale}")

            vel_diff = res_vels - ref_vels
            vel_loss = self._robust_loss(vel_diff, vel_robust_scale)
            vel_loss = jnp.sum(vel_loss, axis=-1)

            # Apply cutoff and snapshot weighting
            if self.cutoff_quantile is not None:
                cutoff = jnp.quantile(vel_loss, self.cutoff_quantile)
                vel_loss = jnp.where(vel_loss < cutoff, vel_loss, 0.0)

            if self.w_snapshot > 0.0:
                vel_loss *= self.w_snapshot

            # Apply reduction
            vel_loss = self._apply_loss_reduction(vel_loss, particle_mean, snapshot_mean)

            if debug:
                print(f"vel_loss = {(self.w_vel * vel_loss):.4e}")

            loss += self.w_vel * vel_loss

        # Two-point statistics loss
        if self.w_cls > 0.0 or self.w_cross > 0.0:
            if res_poss is None:
                raise ValueError("Two-point loss requires res_poss")

            loss += two_point_loss(
                self.mesh_per_dim,
                res_poss,
                ref_cls,
                ref_deltas,
                w_cls=self.w_cls,
                w_cross=self.w_cross,
                w_snapshot=self.w_snapshot,
                k_max=self.k_max,
                k_type=self.k_type,
                eps=self.eps,
                debug=debug,
            )

        # Pressure loss
        if self.w_P > 0.0:
            if res_Ps is None or ref_Ps is None:
                raise ValueError("Pressure loss requires both res_Ps and ref_Ps")
            print(f"w_P = {self.w_P}")

            # Standardize pressure fields
            res_Ps_norm = (res_Ps - jnp.mean(res_Ps)) / (jnp.std(res_Ps) + self.eps)
            ref_Ps_norm = (ref_Ps - jnp.mean(ref_Ps)) / (jnp.std(ref_Ps) + self.eps)
            P_loss = jnp.mean((res_Ps_norm - ref_Ps_norm) ** 2)

            if debug:
                print(f"P_loss = {(self.w_P * P_loss):.4e}")

            loss += self.w_P * P_loss

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
        k_max: Optional[float] = None,
        k_type: str = "step",
        use_arcsinh=False,
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
        self.k_max = k_max
        self.k_type = k_type
        self.use_arcsinh = use_arcsinh
        self.eps = eps

    def __call__(
        self,
        res_poss: jnp.ndarray,
        res_vels: jnp.ndarray,
        ref_poss: Optional[jnp.ndarray] = None,
        ref_vels: Optional[jnp.ndarray] = None,
        ref_cls: Optional[jnp.ndarray] = None,
        ref_deltas: Optional[jnp.ndarray] = None,
        snapshot_mean: bool = True,
        field_mean: bool = True,
        debug: bool = False,
    ):
        print("using field loss")

        loss = 0.0

        # Rho loss
        if self.w_field > 0.0:
            print(f"w_field = {self.w_field}")

            res_rhos = vcic_paint(jnp.zeros([self.mesh_per_dim] * 3), res_poss, 1)
            res_deltas = res_rhos / res_rhos.mean() - 1

            if self.use_arcsinh:
                field_loss = (jnp.arcsinh(res_deltas) - jnp.arcsinh(ref_deltas)) ** 2
            else:
                field_loss = (res_deltas - ref_deltas) ** 2

            if snapshot_mean:
                field_loss = jnp.mean(field_loss, axis=0)
            if field_mean:
                field_loss = jnp.mean(field_loss)

            if debug:
                print(f"field_loss = {(self.w_field * field_loss):.4e}")

            loss += self.w_field * field_loss

        # Two-point statistics loss
        if self.w_cls > 0.0 or self.w_cross > 0.0:
            if res_poss is None:
                raise ValueError("Two-point loss requires res_poss")

            loss += two_point_loss(
                self.mesh_per_dim,
                res_poss,
                ref_cls,
                ref_deltas,
                w_cls=self.w_cls,
                w_cross=self.w_cross,
                w_snapshot=self.w_snapshot,
                k_max=self.k_max,
                k_type=self.k_type,
                eps=self.eps,
                debug=debug,
            )

        return loss
