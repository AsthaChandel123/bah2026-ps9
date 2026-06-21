"""aokit.zernike -- Noll-indexed Zernike polynomials, gradients, index helpers.

STUB: documented signatures raising NotImplementedError. Conventions per
research/03 S1 (Noll single index starting at j=1=piston, RMS/Noll
normalisation so each coefficient is the RMS wavefront of that mode).

Key relations to implement:
  - zern_index(j) -> (n, m)               [research/03 S1.2 aotools zernIndex]
  - radial_poly R_n^m(rho)                [research/03 S1.1]
  - zernike(j, rho, theta)                [Noll-normalised value]
  - zernike_gradient(j, ...)              [analytic d/dx, d/dy via Gamma matrices]
  - noll_covariance / noll_variance c_j   [research/03 S3.2; for r0 fitting]
"""
from __future__ import annotations

from typing import Tuple
import numpy as np


def zern_index(j: int) -> Tuple[int, int]:
    """Map a Noll single index ``j`` (>=1) to ``(n, m)``.

    Even j -> +m (cos), odd j -> -m (sin); j=1 is piston (0,0).
    (research/03 S1.2.)
    """
    raise NotImplementedError("TODO(impl): Noll j -> (n, m)")


def noll_from_nm(n: int, m: int) -> int:
    """Inverse of :func:`zern_index`: ``(n, m) -> j`` (Noll)."""
    raise NotImplementedError("TODO(impl): (n, m) -> Noll j")


def radial_poly(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    """Radial polynomial ``R_n^m(rho)`` (research/03 S1.1)."""
    raise NotImplementedError("TODO(impl): R_n^m radial polynomial")


def zernike(j: int, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Noll-normalised Zernike mode ``Z_j`` on the unit disc (RMS=1)."""
    raise NotImplementedError("TODO(impl): Noll-normalised Z_j")


def zernike_array(j_max: int, n_grid: int, pupil_mask: np.ndarray | None = None
                  ) -> np.ndarray:
    """Stack of Zernike modes Z_2..Z_{j_max} sampled on an ``n_grid`` x
    ``n_grid`` disc, returned as (n_modes, n_pix) over valid pupil pixels.
    Used to build the synthesis matrix ``Z`` (W = Z a)."""
    raise NotImplementedError("TODO(impl): zernike basis array")


def zernike_gradient(j: int, rho: np.ndarray, theta: np.ndarray
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Analytic Cartesian gradients ``(dZ_j/dx, dZ_j/dy)`` on the unit disc.

    Implement via the Noll derivative (Gamma_x, Gamma_y) relations -- the
    derivative of a Zernike is a linear combination of lower-order Zernikes
    (research/03 S2, aotools makegammas). Drives the modal interaction matrix.
    """
    raise NotImplementedError("TODO(impl): analytic Zernike gradients")


def make_gammas(n_radial: int) -> np.ndarray:
    """Noll derivative matrices, shape (2, nmax, nmax) = [Gamma_x, Gamma_y]
    (research/03 S2a; cross-check for the interaction matrix)."""
    raise NotImplementedError("TODO(impl): makegammas Gamma_x/Gamma_y")


def noll_variance(j: int) -> float:
    """Per-mode Kolmogorov variance constant ``c_j`` such that
    ``<a_j^2> = c_j (D/r0)^(5/3)`` (research/03 S3.2, research/04 R1)."""
    raise NotImplementedError("TODO(impl): Noll c_j coefficient")


def noll_residual(j: int) -> float:
    """Residual variance ``Delta_J`` after correcting the first J modes,
    in units of (D/r0)^(5/3) (Noll table; research/03 S3.1)."""
    raise NotImplementedError("TODO(impl): Noll Delta_J residual")
