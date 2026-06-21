"""aokit.turbulence -- multi-method r0 / tau0 estimation (offline).

STUB. >=7 independent r0 estimators (R1-R7) and >=6 tau0 estimators (T1-T6)
across three data domains (raw slopes, reconstructed phase, intensity), plus a
combiner that reports a median +/- spread. research/04 (full report).

Bias removal (mandatory, research/04 S3.3): subtract centroid-noise variance
(tau=0 ACF jump / high-f PSD floor); iterate out modal cross-coupling; exclude
tip/tilt from the r0 fit; use von Karman when L0 is finite.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np


# ============================= r0 ESTIMATORS =============================

def r0_from_zernike_variance(coeffs_ts: np.ndarray, D_m: float,
                             modes: Tuple[int, int] = (4, 15),
                             L0_m: Optional[float] = None,
                             noise_var: Optional[np.ndarray] = None) -> float:
    """R1 (PRIMARY): fit measured Zernike-coefficient variances to
    ``<a_j^2> = c_j (D/r0)^(5/3)`` over mid-order modes (exclude tip/tilt).
    research/04 R1."""
    raise NotImplementedError("TODO(impl): R1 Zernike-variance r0")


def r0_from_slope_variance(slopes_ts: np.ndarray, wavelength_m: float,
                           subap_diam_m: float,
                           noise_var: float = 0.0) -> float:
    """R2: slope (G-tilt) variance r0, ``<alpha^2> = 0.170 lambda^2 r0^-5/3
    d^-1/3`` (reconstruction-independent). research/04 R2."""
    raise NotImplementedError("TODO(impl): R2 slope-variance r0")


def r0_from_dimm(centroids_pair_ts: np.ndarray, baseline_m: float,
                 subap_diam_m: float, wavelength_m: float) -> float:
    """R3: DIMM differential tip/tilt between sub-aperture pairs
    (vibration-immune); longitudinal & transverse averaged. research/04 R3."""
    raise NotImplementedError("TODO(impl): R3 DIMM r0")


def r0_from_phase_variance(phase_ts: np.ndarray, D_m: float,
                           tt_removed: bool = True) -> float:
    """R4: total phase variance ``sigma^2 = 1.0299 (D/r0)^(5/3)``
    (TT-removed: 0.134). research/04 R4."""
    raise NotImplementedError("TODO(impl): R4 phase-variance r0")


def r0_from_structure_function(phase_ts: np.ndarray, sample_pitch_m: float
                               ) -> Tuple[float, float]:
    """R5: Kolmogorov structure-function fit ``D_phi(r)=6.88 (r/r0)^(5/3)``;
    returns (r0, fitted_slope) -- slope ~5/3 validates Kolmogorov. research/04 R5."""
    raise NotImplementedError("TODO(impl): R5 structure-function r0")


def r0_l0_from_vonkarman(coeffs_ts: np.ndarray, D_m: float,
                         modes: Tuple[int, int] = (4, 15)) -> Tuple[float, float]:
    """R6: von Karman joint (r0, L0) fit to the modal-variance curve.
    research/04 R6."""
    raise NotImplementedError("TODO(impl): R6 von Karman (r0, L0)")


def r0_from_seeing(fwhm_rad: float, wavelength_m: float) -> float:
    """R7: seeing FWHM relation ``epsilon = 0.98 lambda / r0`` (image domain).
    research/04 R7."""
    raise NotImplementedError("TODO(impl): R7 seeing r0")


# ============================ tau0 ESTIMATORS ===========================

def tau0_from_autocorr(series_ts: np.ndarray, dt_s: float) -> float:
    """T1 (PRIMARY): 1/e point of the temporal autocorrelation of mid-order
    modes; also isolates noise at tau=0. research/04 T1."""
    raise NotImplementedError("TODO(impl): T1 autocorrelation tau0")


def tau0_from_psd(series_ts: np.ndarray, dt_s: float, D_m: float,
                  radial_orders: Optional[np.ndarray] = None
                  ) -> Tuple[float, float]:
    """T2: temporal PSD cutoff ``f_c ~ 0.3(n+1) v/D`` + slope checks
    (-11/3 tilt, -17/3 higher) -> v -> tau0. research/04 T2."""
    raise NotImplementedError("TODO(impl): T2 PSD tau0")


def tau0_from_greenwood(r0_m: float, wind_speed_mps: float
                        ) -> Tuple[float, float]:
    """T3: Greenwood bridge ``f_G = 0.426 v/r0``, ``tau0 = 0.134/f_G =
    0.314 r0/v``; returns (tau0, f_G). research/04 T3."""
    raise NotImplementedError("TODO(impl): T3 Greenwood tau0")


def tau0_from_structure_function(series_ts: np.ndarray, dt_s: float,
                                 r0_m: float) -> float:
    """T4: temporal structure function ``D_phi(tau)=6.88(v tau/r0)^(5/3)``;
    t0 where D=1 rad^2, tau0 = t0/0.66. research/04 T4."""
    raise NotImplementedError("TODO(impl): T4 temporal structure-function tau0")


def wind_from_frozen_flow(slope_maps_ts: np.ndarray, dt_s: float,
                          sample_pitch_m: float) -> Tuple[float, float]:
    """T5: Taylor frozen-flow wind retrieval -- spatio-temporal cross-correlation
    peak at r = v*tau. Returns (wind_speed_mps, |peak_corr|). research/04 T5."""
    raise NotImplementedError("TODO(impl): T5 frozen-flow wind")


def tyler_frequency(r0_m: float, wind_speed_mps: float, D_m: float) -> float:
    """T6: Tyler tracking frequency ``f_T = 0.368 v r0^-1/6 D^-5/6`` (tip/tilt
    timescale). research/04 T6."""
    raise NotImplementedError("TODO(impl): T6 Tyler frequency")


# =============================== COMBINER ===============================

@dataclass
class TurbulenceResult:
    r0_m: Dict[str, float] = field(default_factory=dict)      # estimator -> value
    tau0_s: Dict[str, float] = field(default_factory=dict)
    L0_m: Optional[float] = None
    wind_speed_mps: Optional[float] = None
    f_greenwood_hz: Optional[float] = None
    seeing_arcsec: Optional[float] = None
    strehl_marechal: Optional[float] = None
    r0_median: Optional[float] = None
    r0_spread: Optional[float] = None
    tau0_median: Optional[float] = None
    tau0_spread: Optional[float] = None
    n_frames: int = 0
    dt_s: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict:
        """Serialise to the turbulence_summary.json schema (ARCHITECTURE.md S4.4)."""
        raise NotImplementedError("TODO(impl): TurbulenceResult.to_dict")


def combine_estimates(values: Dict[str, float]) -> Tuple[float, float]:
    """Robust central estimate (median) and spread (e.g. MAD or std) over a set
    of independent estimator values. research/04 S5."""
    raise NotImplementedError("TODO(impl): median + spread combiner")


def estimate_all(slopes_ts: np.ndarray, coeffs_ts: np.ndarray,
                 phase_ts: Optional[np.ndarray], cfg, dt_s: float
                 ) -> TurbulenceResult:
    """Run every applicable r0/tau0 estimator, remove biases, combine, and
    return a fully populated :class:`TurbulenceResult`. research/04 S5 pipeline."""
    raise NotImplementedError("TODO(impl): full multi-method estimate_all")
