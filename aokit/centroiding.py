"""aokit.centroiding -- reference centroiders for validation.

STUB. Python references for CoG / TCoG / WCoG / TWCoG / IWCoG / brightest-pixel
/ correlation / Gaussian-fit, used to validate the C TWCoG kernel and to serve
as ground-truth oracles (Gaussian-fit / ML are offline references). research/01.

All functions operate on a 2-D sub-aperture window and return a sub-pixel
``(cx, cy)`` in window-local coordinates unless noted.
"""
from __future__ import annotations

from typing import Tuple, Optional
import numpy as np


def cog(window: np.ndarray) -> Tuple[float, float]:
    """Plain center-of-gravity (first moment). research/01 M1."""
    raise NotImplementedError("TODO(impl): plain CoG")


def thresholded_cog(window: np.ndarray, thresh: float) -> Tuple[float, float]:
    """Thresholded CoG: subtract ``thresh``, clip to 0, then CoG. research/01 M2."""
    raise NotImplementedError("TODO(impl): thresholded CoG")


def weighted_cog(window: np.ndarray, weights: np.ndarray) -> Tuple[float, float]:
    """Weighted CoG with a (precomputed) Gaussian weight. research/01 M3."""
    raise NotImplementedError("TODO(impl): weighted CoG")


def twcog(window: np.ndarray, weights: Optional[np.ndarray],
          thresh_frac: float = 0.05, thresh_sigma: float = 0.0,
          gain: float = 1.0, min_pixels: int = 3) -> Tuple[float, float]:
    """Thresholded + Windowed Weighted CoG (the PRIMARY estimator; the C core
    mirrors this). research/01 S12."""
    raise NotImplementedError("TODO(impl): TWCoG (primary)")


def iter_weighted_cog(window: np.ndarray, sigma_w: float,
                      n_iter: int = 2) -> Tuple[float, float]:
    """Iteratively-weighted CoG (weight re-centred each iter). research/01 M4."""
    raise NotImplementedError("TODO(impl): iteratively-weighted CoG")


def brightest_pixel(window: np.ndarray, n_bright: int) -> Tuple[float, float]:
    """Brightest-pixel selection then CoG (Basden). research/01 M7."""
    raise NotImplementedError("TODO(impl): brightest-pixel centroiding")


def correlation_centroid(window: np.ndarray, template: np.ndarray
                         ) -> Tuple[float, float]:
    """Correlation/matched-filter centroid (extended/elongated spots).
    research/01 M10. Includes a peak-locking-aware sub-pixel step."""
    raise NotImplementedError("TODO(impl): correlation centroiding")


def gaussfit_centroid(window: np.ndarray) -> Tuple[float, float]:
    """2-D Gaussian least-squares fit centroid (offline ground truth).
    research/01 M9."""
    raise NotImplementedError("TODO(impl): Gaussian-fit centroid")


def gaussian_weight(win_w: int, win_h: int, cx: float, cy: float,
                    fwhm_px: float) -> np.ndarray:
    """Precompute a Gaussian weight window centred at (cx, cy) for WCoG/TWCoG."""
    raise NotImplementedError("TODO(impl): Gaussian weight LUT")
