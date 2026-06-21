"""aokit.datagen -- synthetic SH-WFS data generation (the validation backbone).

STUB. Generates phase screens (>=3 methods), spot fields (geometric + Fraunhofer),
detector noise, and frozen-flow time-series with KNOWN injected r0/tau0, then
writes .bmp frames + a ground-truth config. research/07 PART B.

Pipeline: phase screen phi(x,y) [known r0, L0] -> propagate through MLA
(geometric tilt or Fraunhofer) -> spot field -> detector (Poisson + read +
quantize) -> .bmp; translate screen by v*dt for the next frame (frozen flow =>
known tau0 = 0.314 r0 / v).
"""
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from .config import Config


# --------------------------- phase-screen generators ---------------------

def ft_phase_screen(r0_m: float, N: int, delta_m: float,
                    L0_m: float = np.inf, l0_m: float = 0.0,
                    seed: Optional[int] = None) -> np.ndarray:
    """Method 1: FFT/spectral phase screen from the (modified) von Karman PSD
    ``Phi = 0.023 r0^-5/3 exp(-(f/fm)^2) / (f^2 + f0^2)^(11/6)``. research/07 B.1 M1."""
    raise NotImplementedError("TODO(impl): FFT phase screen")


def ft_sh_phase_screen(r0_m: float, N: int, delta_m: float,
                       L0_m: float = np.inf, l0_m: float = 0.0,
                       n_subharmonics: int = 3,
                       seed: Optional[int] = None) -> np.ndarray:
    """Method 2 (RECOMMENDED for single frames): FFT + subharmonics to augment
    low-frequency (tip/tilt) content. research/07 B.1 M2."""
    raise NotImplementedError("TODO(impl): FFT + subharmonics phase screen")


def zernike_phase_screen(r0_m: float, D_m: float, N: int, j_max: int,
                         seed: Optional[int] = None) -> np.ndarray:
    """Method 3: Zernike synthesis with Noll-covariance-drawn coefficients
    (deterministic unit tests). research/07 B.1 M3."""
    raise NotImplementedError("TODO(impl): Zernike phase screen")


def covariance_phase_screen(r0_m: float, N: int, delta_m: float,
                            L0_m: float = np.inf,
                            seed: Optional[int] = None) -> np.ndarray:
    """Method 4: Cholesky of the von Karman covariance (statistical gold
    standard). research/07 B.1 M4."""
    raise NotImplementedError("TODO(impl): Cholesky covariance phase screen")


# ----------------------------- spot-field synthesis ----------------------

def spots_geometric(cfg: Config, phase: np.ndarray) -> np.ndarray:
    """Model 1: geometric tilt-per-sub-aperture. Mean local slope tilts each
    spot by Delta = f*theta; place a diffraction-limited spot at ref + Delta.
    This is the analytic oracle (inverse of the centroiding model). research/07 B.2 M1."""
    raise NotImplementedError("TODO(impl): geometric spot field")


def spots_fraunhofer(cfg: Config, phase: np.ndarray) -> np.ndarray:
    """Model 2: per-sub-aperture Fraunhofer/FFT diffraction,
    ``I = |FFT(U_sub * mask)|^2``, tiled to the detector. Physically faithful
    (spot broadening/speckle). research/07 B.2 M2."""
    raise NotImplementedError("TODO(impl): Fraunhofer spot field")


def apply_detector_noise(image: np.ndarray, flux_photons: float,
                         qe: float = 0.9, read_noise_e: float = 3.0,
                         gain: float = 1.0, bias: float = 0.0,
                         bit_depth: int = 8,
                         seed: Optional[int] = None) -> np.ndarray:
    """Poisson shot + Gaussian read + quantization to ``bit_depth``. research/07 B.3."""
    raise NotImplementedError("TODO(impl): detector noise model")


# ------------------------------ time-series ------------------------------

def frozen_flow_series(cfg: Config, r0_m: float, wind_speed_mps: float,
                       wind_angle_deg: float, n_frames: int,
                       L0_m: float = np.inf, boiling: float = 0.0,
                       seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
    """Generate a frozen-flow phase-screen series translated by v*dt per frame
    (known tau0 = 0.314 r0 / v). Optional AR "boiling" to stress-test the
    frozen-flow assumption. Returns (phase_series[T,N,N], ground_truth_dict).
    research/07 B.5."""
    raise NotImplementedError("TODO(impl): frozen-flow phase-screen series")


def generate_dataset(cfg: Config, r0_m: float, tau0_s: float, n_frames: int,
                     out_dir: str, spot_model: str = "geometric",
                     seed: Optional[int] = None) -> dict:
    """End-to-end: build the frozen-flow series, synthesize spot fields, add
    noise, write .bmp frames + ground-truth JSON into ``out_dir``. Returns the
    ground-truth dict (injected r0/tau0/wind/L0). research/07 B, S B.4."""
    raise NotImplementedError("TODO(impl): full synthetic dataset generation")
