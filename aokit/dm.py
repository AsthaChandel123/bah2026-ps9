"""aokit.dm -- DM influence matrix H, command matrix H+, stroke handling.

STUB. Builds the Gaussian influence-function matrix H (with the provided
inter-actuator coupling c), the regularized command matrix
``G = H+ * (-1/2)`` (coupling DECONVOLVED, reflection factor included), and the
stroke-unit conversion. research/05 S1-S4, S7, S9, S11.
"""
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from .config import Config
from .geometry import ActuatorGrid


def gaussian_sigma_from_coupling(pitch_m: float, coupling: float) -> float:
    """Gaussian IF width from coupling: ``sigma = d / sqrt(-2 ln c)`` so the
    bump drops to ``c`` at the actuator pitch. research/05 S1.3 (KEY relation)."""
    raise NotImplementedError("TODO(impl): sigma = d/sqrt(-2 ln c)")


def influence_function(r: np.ndarray, pitch_m: float, coupling: float,
                       model: str = "gaussian", alpha: float = 2.0) -> np.ndarray:
    """Evaluate IF(r): Gaussian ``exp(-r^2/2 sigma^2)`` (alpha=2) or the
    power-law ``exp(ln(c) (r/d)^alpha)``. research/05 S1.3-1.4."""
    raise NotImplementedError("TODO(impl): influence function model")


def build_influence_matrix(cfg: Config, acts: ActuatorGrid,
                           sample_xy: np.ndarray) -> np.ndarray:
    """Influence matrix H (N_pts x N_act): column a is actuator a's IF sampled
    at ``sample_xy`` (e.g. the phase grid). Surface = H @ a. research/05 S2."""
    raise NotImplementedError("TODO(impl): influence matrix H")


def build_command_matrix(H: np.ndarray, method: str = "tikhonov",
                         tikhonov_mu: float = 1e-3,
                         sv_threshold: float = 1e-6,
                         reflection_factor: float = -0.5,
                         remove_waffle: bool = True) -> np.ndarray:
    """Regularized command matrix ``G = (H^T H + mu^2 I)^-1 H^T * reflection``
    (or truncated-SVD), so ``a = G @ phi = H+ (-phi/2)``. Coupling is baked into
    H, so G contains the anti-coupling off-diagonals. research/05 S3, S4, S9."""
    raise NotImplementedError("TODO(impl): regularized DM command matrix G")


def fuse_slopes_to_commands(G: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Fused slopes->commands matrix K = G @ R (single MVM path). research/02 S12."""
    raise NotImplementedError("TODO(impl): fuse K = G @ R")


def clip_strokes(commands: np.ndarray, stroke_max: float) -> Tuple[np.ndarray, int]:
    """Clip commands to +/- stroke_max; return (clipped, n_saturated).
    research/05 S4.3."""
    raise NotImplementedError("TODO(impl): stroke clipping")


def to_stroke_units(commands: np.ndarray, gain_m_per_unit: float) -> np.ndarray:
    """Convert unit commands to physical stroke length (m) via gain g.
    research/05 S7."""
    raise NotImplementedError("TODO(impl): stroke-unit conversion")


def build_interaction_matrix(poke_slopes: np.ndarray, poke_amp: float
                             ) -> np.ndarray:
    """Calibration-based interaction matrix D = d slopes / d command from poke
    frames (alternative to the model H). research/05 S6."""
    raise NotImplementedError("TODO(impl): calibration interaction matrix D")
