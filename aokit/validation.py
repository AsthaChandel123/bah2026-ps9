"""aokit.validation -- correctness & quality metrics (recovered vs. true).

STUB. RMS WFE, Strehl (Marechal), phase correlation, r0/tau0 recovery error,
reconstructor self-consistency, DM-corrected residual, C/Python parity.
research/07 PART C.
"""
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np


def rms_wfe(W_recon: np.ndarray, W_true: np.ndarray,
            mask: Optional[np.ndarray] = None,
            remove_piston: bool = True, remove_tilt: bool = True) -> float:
    """RMS wavefront error over the valid pupil (piston/tilt removed).
    research/07 C.1."""
    raise NotImplementedError("TODO(impl): RMS WFE")


def strehl_marechal(residual_phase_rad: np.ndarray,
                    mask: Optional[np.ndarray] = None) -> float:
    """Strehl via the Marechal approximation ``S = exp(-sigma_phi,res^2)``
    (sigma in rad). research/07 C.1."""
    raise NotImplementedError("TODO(impl): Strehl (Marechal)")


def phase_correlation(W_recon: np.ndarray, W_true: np.ndarray,
                      mask: Optional[np.ndarray] = None) -> float:
    """Pearson correlation between reconstructed and true phase over the pupil.
    research/07 C.1."""
    raise NotImplementedError("TODO(impl): phase correlation")


def r0_recovery_error(r0_est: float, r0_true: float) -> float:
    """Relative r0 recovery error ``|r0_est - r0_true| / r0_true``. research/07 C.1."""
    raise NotImplementedError("TODO(impl): r0 recovery error")


def tau0_recovery_error(tau0_est: float, tau0_true: float) -> float:
    """Relative tau0 recovery error. research/07 C.1."""
    raise NotImplementedError("TODO(impl): tau0 recovery error")


def dm_residual(W_true: np.ndarray, dm_surface: np.ndarray,
                mask: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """Residual after DM correction: ``W_res = W_true - dm_surface``; returns
    (rms_residual, strehl). research/07 C.1."""
    raise NotImplementedError("TODO(impl): DM-corrected residual")


def reconstructor_self_consistency(slopes_meas: np.ndarray,
                                   W_recon: np.ndarray, Gamma: np.ndarray
                                   ) -> float:
    """Round-trip residual ``||Gamma @ phi_recon - slopes_meas||`` (should be
    ~0). research/07 C.2."""
    raise NotImplementedError("TODO(impl): reconstructor self-consistency")


def cpython_parity(slopes_c: np.ndarray, slopes_py: np.ndarray) -> float:
    """Max abs difference between C-core and Python-reference slopes for the
    same frame (guards the fast path). research/07 C.2."""
    raise NotImplementedError("TODO(impl): C/Python parity metric")


def noll_variance_check(coeffs_ts: np.ndarray, D_m: float, r0_true: float
                        ) -> np.ndarray:
    """Monte-Carlo check: ensemble Zernike variances should follow Noll
    ``c_j (D/r0)^(5/3)``; returns per-mode ratio (measured/expected). research/07 C.4."""
    raise NotImplementedError("TODO(impl): Noll-variance Monte-Carlo check")
