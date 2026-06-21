"""aokit.reconstructor -- build zonal + modal reconstructors (offline).

STUB. Builds:
  - the Fried gradient matrix Gamma and the regularized zonal reconstructor R
    (SVD/Tikhonov, piston + waffle removed)        research/02 S3, S4, S6, S15
  - the modal interaction matrix M, reconstructor M+ (SVD/Tikhonov) and the
    Zernike synthesis matrix Z                      research/03 S2-S4, S6
  - an FFT Fourier Transform Reconstructor (Fried filter) as a cross-check /
    scaling fallback                                research/02 S7

These matrices are serialized to AOMX for the C core (scripts/build_calibration).
"""
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from .config import Config
from .geometry import SubApertureGrid


# ----------------------------- ZONAL (Fried) -----------------------------

def build_fried_gamma(cfg: Config, grid: SubApertureGrid) -> np.ndarray:
    """Fried gradient matrix Gamma (2M x N) over valid sub-apertures, using the
    corner-averaging equations
        s_x = (1/2h)[(phi_b - phi_a) + (phi_d - phi_c)]
        s_y = (1/2h)[(phi_c - phi_a) + (phi_d - phi_b)]
    research/02 S2.2, S4."""
    raise NotImplementedError("TODO(impl): Fried Gamma matrix")


def waffle_vector(n_act_x: int, n_act_y: int) -> np.ndarray:
    """The checkerboard waffle null-mode vector on the corner grid. research/02 S6."""
    raise NotImplementedError("TODO(impl): waffle mode vector")


def build_zonal_reconstructor(cfg: Config, grid: SubApertureGrid,
                              method: str = "svd",
                              sv_threshold: float = 1e-6,
                              tikhonov_mu: float = 0.0,
                              remove_waffle: bool = True,
                              remove_piston: bool = True,
                              r0_m: Optional[float] = None) -> np.ndarray:
    """Regularized least-squares Fried reconstructor R (N x 2M) such that
    ``phi = R @ s``. SVD with singular-value thresholding (or Tikhonov), then
    explicit piston + waffle nulling. Optionally fold a Kolmogorov/MMSE prior
    via ``r0_m``. research/02 S3, S6, S10, S15."""
    raise NotImplementedError("TODO(impl): zonal Fried LS reconstructor R")


# ------------------------------- MODAL (Zernike) -------------------------

def build_modal_interaction(cfg: Config, grid: SubApertureGrid,
                            j_max: int) -> np.ndarray:
    """Modal interaction matrix M (2M x J): column j is the SH slope response
    to Zernike mode j (analytic sub-aperture-averaged gradients). research/03 S2."""
    raise NotImplementedError("TODO(impl): modal interaction matrix M")


def build_modal_reconstructor(M: np.ndarray, method: str = "tikhonov",
                              sv_threshold: float = 1e-6,
                              tikhonov_mu: float = 1e-3) -> np.ndarray:
    """Modal reconstructor M+ (J x 2M), ``a = M+ @ s``, via SVD with Tikhonov
    damping or truncated SVD. research/03 S2.4."""
    raise NotImplementedError("TODO(impl): modal reconstructor M+")


def build_synthesis_matrix(cfg: Config, j_max: int, n_grid: int,
                           pupil_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Zernike synthesis matrix Z (N_pts x J): ``W = Z @ a``. research/03 S0."""
    raise NotImplementedError("TODO(impl): Zernike synthesis matrix Z")


# ------------------------- FFT reconstructor (validation) ----------------

def ftr_fried_filters(shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Exact Fried FFT filters (gx, gy) with Nyquist rows/cols zeroed (= waffle
    removal):
        gx = (exp(i fy) + 1) * (exp(i fx) - 1)
        gy = (exp(i fx) + 1) * (exp(i fy) - 1)
    research/02 S7."""
    raise NotImplementedError("TODO(impl): Fried FTR filters")


def ftr_reconstruct(slopes_x: np.ndarray, slopes_y: np.ndarray
                    ) -> np.ndarray:
    """Fourier Transform Reconstructor (Fried): FFT slopes, divide by the
    least-squares Fried filter, inverse-FFT; set piston (0,0)=0. Cross-check /
    scaling fallback for the dense R. research/02 S7."""
    raise NotImplementedError("TODO(impl): FFT Fried reconstructor")


def integrate_slopes(slopes_x: np.ndarray, slopes_y: np.ndarray
                     ) -> np.ndarray:
    """Direct path integration baseline (cumulative trapezoid). research/02 S5."""
    raise NotImplementedError("TODO(impl): direct slope integration")
