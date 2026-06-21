"""aokit.geometry -- Fried sensor/actuator geometry & calibration.

STUB. Builds the sub-aperture grid, the (n+1)x(n+1) actuator-corner grid, the
pupil mask, the active-sub-aperture (flux) mask, reference spot positions, and
the lenslet<->actuator index maps. research/02 S1, S5; research/01 S4-S5;
research/07 A.4.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from .config import Config


@dataclass
class SubApertureGrid:
    """Per-lenslet window geometry + reference centroids + validity mask."""
    x0: np.ndarray        # (n_sub,) window top-left x (px)
    y0: np.ndarray        # (n_sub,) window top-left y (px)
    w: int                # window width (px)
    h: int                # window height (px)
    ref_x: np.ndarray     # (n_sub,) reference centroid x (px)
    ref_y: np.ndarray     # (n_sub,) reference centroid y (px)
    valid: np.ndarray     # (n_sub,) bool active mask
    ij: np.ndarray        # (n_sub, 2) lenslet (col,row) indices


@dataclass
class ActuatorGrid:
    """Fried actuator-corner grid."""
    x: np.ndarray         # (n_act,) actuator x on pupil (px or m)
    y: np.ndarray         # (n_act,) actuator y
    ij: np.ndarray        # (n_act, 2) (col,row) indices on the (n+1) grid


def build_subaperture_grid(cfg: Config) -> SubApertureGrid:
    """Nominal sub-aperture windows from MLA pitch / pixel size / pupil center.
    Window size ~ pixels-per-lenslet; centers tile the pupil. research/01 S4."""
    raise NotImplementedError("TODO(impl): nominal sub-aperture grid")


def register_references(cfg: Config, flat_frame: np.ndarray,
                        grid: Optional[SubApertureGrid] = None
                        ) -> SubApertureGrid:
    """Detect the spot in each cell of a flat-wavefront frame, CoG it, and set
    reference centroids + cell origins (registration). research/01 S5."""
    raise NotImplementedError("TODO(impl): reference-spot registration")


def active_subaperture_mask(cfg: Config, flat_frame: np.ndarray,
                            grid: SubApertureGrid,
                            flux_frac: float = 0.5) -> np.ndarray:
    """Boolean mask of illuminated sub-apertures (flux >= flux_frac of the
    unobscured-cell flux). research/01 S4."""
    raise NotImplementedError("TODO(impl): active-subaperture flux mask")


def pupil_mask(cfg: Config, n_grid: int) -> np.ndarray:
    """Circular pupil mask on an ``n_grid`` x ``n_grid`` array (True inside)."""
    raise NotImplementedError("TODO(impl): circular pupil mask")


def build_actuator_grid(cfg: Config) -> ActuatorGrid:
    """Fried (n+1)x(n+1) actuator-corner grid aligned to the lenslet grid.
    research/02 S12; research/05 S5."""
    raise NotImplementedError("TODO(impl): Fried actuator-corner grid")


def lenslet_actuator_maps(grid: SubApertureGrid, acts: ActuatorGrid
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """Index maps: for each lenslet, the 4 surrounding actuator-corner indices
    (and the inverse). Fried adjacency. research/02 S12, research/05 S5."""
    raise NotImplementedError("TODO(impl): lenslet<->actuator index maps")


def fried_corner_indices(n_lenslets_x: int, n_lenslets_y: int) -> np.ndarray:
    """For each sub-aperture, the (a,b,c,d)=(TL,TR,BL,BR) corner-phase indices
    on the (n+1)x(n+1) grid, row-major. Used to build the Fried Gamma matrix.
    research/02 S4."""
    raise NotImplementedError("TODO(impl): Fried (a,b,c,d) corner indices")
