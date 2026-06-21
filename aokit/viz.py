"""aokit.viz -- matplotlib visualizations (offline).

STUB. Plots: spot field, reconstructed phase maps, Zernike spectrum, r0/tau0
trends, residuals after DM correction. research/06 S13 (offline visualization).

Imported optionally by aokit.__init__ (matplotlib is an optional dependency);
functions use a non-interactive backend safe for headless runs.
"""
from __future__ import annotations

from typing import Optional, Sequence
import numpy as np


def plot_spot_field(image: np.ndarray, subaps=None, centroids=None,
                    title: str = "SH-WFS spot field", save: Optional[str] = None):
    """Show the detector frame with optional sub-aperture grid + centroids."""
    raise NotImplementedError("TODO(impl): plot spot field")


def plot_phase_map(W: np.ndarray, mask: Optional[np.ndarray] = None,
                   title: str = "Reconstructed wavefront",
                   save: Optional[str] = None):
    """Display a reconstructed phase map (rad)."""
    raise NotImplementedError("TODO(impl): plot phase map")


def plot_zernike_spectrum(coeffs: np.ndarray,
                          title: str = "Zernike spectrum",
                          save: Optional[str] = None):
    """Bar plot of Zernike coefficients (Noll index)."""
    raise NotImplementedError("TODO(impl): plot Zernike spectrum")


def plot_turbulence_trends(r0_ts: Sequence[float], tau0_ts: Sequence[float],
                           dt_s: float, save: Optional[str] = None):
    """Time trends of r0 and tau0 over the sequence (with estimator spread)."""
    raise NotImplementedError("TODO(impl): plot r0/tau0 trends")


def plot_actuator_map(commands: np.ndarray, n_act_x: int, n_act_y: int,
                      title: str = "DM actuator map (stroke units)",
                      save: Optional[str] = None):
    """2-D image of the actuator-stroke map on the Fried grid."""
    raise NotImplementedError("TODO(impl): plot actuator map")


def plot_residual(W_before: np.ndarray, W_after: np.ndarray,
                  mask: Optional[np.ndarray] = None, save: Optional[str] = None):
    """Side-by-side wavefront before/after DM correction + residual."""
    raise NotImplementedError("TODO(impl): plot DM residual")
