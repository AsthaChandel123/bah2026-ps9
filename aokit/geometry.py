"""aokit.geometry -- Fried sensor/actuator geometry & calibration.

Builds the sub-aperture grid, the (n+1)x(n+1) actuator-corner grid, the pupil
mask, the active-sub-aperture (flux) mask, reference spot positions, and the
lenslet<->actuator index maps. research/02 S1, S5; research/01 S4-S5;
research/07 A.4.

CANONICAL ORDERINGS (the C core and the Python reconstructor BOTH rely on these;
do not change without updating reconstructor.py / dm.py / the C core):

* Coordinate axes (detector pixels): ``x`` is the column axis (increasing
  left->right), ``y`` is the row axis (increasing top->bottom). This is the
  standard image/detector convention. ``ij`` arrays store ``(col, row) = (i, j)``
  where ``i`` indexes x and ``j`` indexes y.

* Lenslet / sub-aperture ordering: lenslets are enumerated **row-major** over the
  ``n_y x n_x`` grid -- ``flat = row * n_x + col`` (row = y index, col = x index,
  both 0-based) -- and the *valid* sub-apertures keep that relative order
  (``SubApertureGrid`` arrays are the valid subset, in increasing flat index).
  This order defines the slope-vector layout below.

* Slope-vector layout (THE contract, ARCHITECTURE.md S2 fast path
  ``s = [sx_1..sx_M, sy_1..sy_M]``): the 2M slope vector is **block** layout --
  all x-slopes for the M valid sub-apertures (in lenslet order above), then all
  y-slopes. NOT interleaved. ``slope_vector_layout()`` documents/returns this.

* Actuator (corner) ordering: the (n_x+1) x (n_y+1) Fried corner nodes are
  enumerated **row-major** -- ``flat = row * (n_x + 1) + col``. This is the phase
  / actuator unknown ordering ``phi = [phi_0 .. phi_{N-1}]`` consumed by the
  Fried Gamma matrix and the DM influence matrix. The four corners of lenslet
  ``(col, row)`` are, as ``(a, b, c, d) = (TL, TR, BL, BR)``:
      a (TL) = (col,   row)     -> row*(n_x+1) + col
      b (TR) = (col+1, row)     -> row*(n_x+1) + col + 1
      c (BL) = (col,   row+1)   -> (row+1)*(n_x+1) + col
      d (BR) = (col+1, row+1)   -> (row+1)*(n_x+1) + col + 1
  matching the Fried corner-averaging equations
      s_x = (1/2h)[(phi_b - phi_a) + (phi_d - phi_c)]
      s_y = (1/2h)[(phi_c - phi_a) + (phi_d - phi_b)]   (research/02 S2.2, S4).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from .config import Config


@dataclass
class SubApertureGrid:
    """Per-lenslet window geometry + reference centroids + validity mask.

    All arrays are length ``n_sub`` and ordered by increasing lenslet flat index
    ``row * n_lenslets_x + col`` (the canonical lenslet order; see module
    docstring). When ``valid`` selection has been applied (the typical case) the
    arrays already contain only the valid sub-apertures, still in that order.
    """
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
    """Fried actuator-corner grid.

    All arrays are length ``n_act = (n_lenslets_x + 1) * (n_lenslets_y + 1)`` and
    ordered row-major (``flat = row * (n_lenslets_x + 1) + col``), the canonical
    phase/actuator-unknown order.
    """
    x: np.ndarray         # (n_act,) actuator x on pupil (px)
    y: np.ndarray         # (n_act,) actuator y (px)
    ij: np.ndarray        # (n_act, 2) (col,row) indices on the (n+1) grid
    # Normalized unit-disk coordinates (for Zernike basis/gradients).
    x_norm: np.ndarray    # (n_act,) actuator x / pupil_radius
    y_norm: np.ndarray    # (n_act,) actuator y / pupil_radius
    valid: np.ndarray     # (n_act,) bool: touches >=1 valid sub-aperture


@dataclass
class Geometry:
    """Bundle of all Fried geometry products for one optical configuration.

    Produced by :func:`build_geometry`; consumed by ``reconstructor.py``,
    ``dm.py``, ``turbulence.py`` and ``scripts/build_calibration.py``.
    """
    cfg: Config
    subaps: SubApertureGrid          # valid sub-apertures only (canonical order)
    acts: ActuatorGrid               # full (n+1)x(n+1) corner grid
    corner_idx: np.ndarray           # (M, 4) per-valid-lenslet (a,b,c,d) corner indices
    lenslet_to_act: np.ndarray       # alias of corner_idx (M, 4)
    act_to_lenslet: np.ndarray       # (n_act, 4) lenslets touching each actuator (-1 pad)
    # Normalized unit-disk coordinates of the valid sub-aperture centers.
    subap_x_norm: np.ndarray         # (M,)
    subap_y_norm: np.ndarray         # (M,)
    pupil_radius_px: float           # pupil radius in detector pixels

    @property
    def n_sub(self) -> int:
        return int(self.subaps.ref_x.shape[0])

    @property
    def n_act(self) -> int:
        return int(self.acts.x.shape[0])

    @property
    def n_slopes(self) -> int:
        return 2 * self.n_sub


# ----------------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------------

def _lenslet_centers_px(cfg: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nominal lenslet-center pixel coordinates for every lenslet on the full
    n_y x n_x grid, in canonical row-major order.

    Returns ``(cx, cy, ij)`` where ``cx, cy`` are length ``n_x*n_y`` and ``ij``
    is ``(n_x*n_y, 2)`` of ``(col, row)``. Honors ``geometry.rotation_deg`` and
    ``geometry.flip_y`` about the pupil center.
    """
    nx, ny = cfg.mla.n_lenslets_x, cfg.mla.n_lenslets_y
    ppl = cfg.px_per_lenslet
    cx0, cy0 = cfg.pupil.center_x_px, cfg.pupil.center_y_px

    cols = np.arange(nx)
    rows = np.arange(ny)
    cc, rr = np.meshgrid(cols, rows)          # row-major: rr varies slowest
    cc = cc.ravel()
    rr = rr.ravel()

    # Offsets from pupil center (center the grid on the pupil).
    dx = (cc - (nx - 1) / 2.0) * ppl
    dy = (rr - (ny - 1) / 2.0) * ppl

    if cfg.geometry.flip_y:
        dy = -dy

    theta = np.deg2rad(cfg.geometry.rotation_deg)
    if theta != 0.0:
        ct, st = np.cos(theta), np.sin(theta)
        dxr = ct * dx - st * dy
        dyr = st * dx + ct * dy
        dx, dy = dxr, dyr

    cx = cx0 + dx
    cy = cy0 + dy
    ij = np.stack([cc, rr], axis=1).astype(np.int64)
    return cx, cy, ij


def _corner_nodes_px(cfg: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pixel coordinates of the (n_x+1)x(n_y+1) Fried corner nodes, row-major.

    Returns ``(ax, ay, ij)``. Built with the same center/rotation/flip transform
    as the lenslet centers so corners land exactly between lenslet centers.
    """
    nx, ny = cfg.mla.n_lenslets_x, cfg.mla.n_lenslets_y
    ppl = cfg.px_per_lenslet
    cx0, cy0 = cfg.pupil.center_x_px, cfg.pupil.center_y_px

    cols = np.arange(nx + 1)
    rows = np.arange(ny + 1)
    cc, rr = np.meshgrid(cols, rows)
    cc = cc.ravel()
    rr = rr.ravel()

    # Corner k sits half a pitch *before* lenslet-center column k, i.e. at the
    # shared corner between lenslets. The corner grid is centered on the pupil.
    dx = (cc - nx / 2.0) * ppl
    dy = (rr - ny / 2.0) * ppl

    if cfg.geometry.flip_y:
        dy = -dy

    theta = np.deg2rad(cfg.geometry.rotation_deg)
    if theta != 0.0:
        ct, st = np.cos(theta), np.sin(theta)
        dxr = ct * dx - st * dy
        dyr = st * dx + ct * dy
        dx, dy = dxr, dyr

    ax = cx0 + dx
    ay = cy0 + dy
    ij = np.stack([cc, rr], axis=1).astype(np.int64)
    return ax, ay, ij


def pupil_radius_px(cfg: Config) -> float:
    """Pupil radius in detector pixels (``diameter_m / pixel_size_m / 2``)."""
    return 0.5 * cfg.pupil.diameter_m / cfg.camera.pixel_size_m


def _cell_fill_fraction(cfg: Config, cx: np.ndarray, cy: np.ndarray,
                        ss: int = 11) -> np.ndarray:
    """Fraction of each (square, side = pitch) sub-aperture cell that falls
    inside the circular pupil, estimated on an ``ss x ss`` sub-pixel grid.

    Used as the *geometric* illumination proxy for the active-sub-aperture mask
    when no flat frame is available.
    """
    ppl = cfg.px_per_lenslet
    R = pupil_radius_px(cfg)
    cx0, cy0 = cfg.pupil.center_x_px, cfg.pupil.center_y_px

    off = (np.arange(ss) - (ss - 1) / 2.0) / ss * ppl   # sub-pixel offsets
    ox, oy = np.meshgrid(off, off)
    ox = ox.ravel()
    oy = oy.ravel()

    frac = np.empty(cx.shape[0], dtype=np.float64)
    R2 = R * R
    for k in range(cx.shape[0]):
        gx = cx[k] + ox
        gy = cy[k] + oy
        inside = (gx - cx0) ** 2 + (gy - cy0) ** 2 <= R2
        frac[k] = float(np.mean(inside))
    return frac


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

def build_subaperture_grid(cfg: Config,
                           flux_frac: float = 0.5,
                           valid_only: bool = True) -> SubApertureGrid:
    """Nominal sub-aperture windows from MLA pitch / pixel size / pupil center.

    Window size ~ pixels-per-lenslet; centers tile the pupil. research/01 S4.

    The pupil-geometry **fill-fraction** of each cell (fraction of the square
    cell area inside the pupil disk) selects which sub-apertures are sufficiently
    illuminated: a cell is *valid* iff ``fill_fraction >= flux_frac``. With
    ``valid_only=True`` (default) the returned grid contains only the valid
    sub-apertures, in canonical lenslet order, so its arrays directly index the
    slope vector. With ``valid_only=False`` all ``n_x*n_y`` cells are returned
    (``valid`` flags which are active) -- useful for diagnostics / plotting.

    A measured flat frame can refine both references and validity afterwards via
    :func:`register_references` and :func:`active_subaperture_mask`.
    """
    cx, cy, ij = _lenslet_centers_px(cfg)
    win = int(round(cfg.px_per_lenslet))
    if win < 1:
        win = 1

    frac = _cell_fill_fraction(cfg, cx, cy)
    valid = frac >= float(flux_frac)

    x0 = cx - win / 2.0
    y0 = cy - win / 2.0

    if valid_only:
        sel = np.where(valid)[0]
        return SubApertureGrid(
            x0=x0[sel].copy(), y0=y0[sel].copy(),
            w=win, h=win,
            ref_x=cx[sel].copy(), ref_y=cy[sel].copy(),
            valid=np.ones(sel.shape[0], dtype=bool),
            ij=ij[sel].copy(),
        )
    return SubApertureGrid(
        x0=x0, y0=y0, w=win, h=win,
        ref_x=cx.copy(), ref_y=cy.copy(),
        valid=valid, ij=ij,
    )


def register_references(cfg: Config, flat_frame: np.ndarray,
                        grid: Optional[SubApertureGrid] = None
                        ) -> SubApertureGrid:
    """Detect the spot in each cell of a flat-wavefront frame, CoG it, and set
    reference centroids + cell origins (registration). research/01 S5.

    For each sub-aperture window of the (nominal or supplied) ``grid``, a plain
    center-of-gravity over the window pixels of ``flat_frame`` gives the measured
    reference centroid; ``ref_x/ref_y`` are overwritten with those values
    (windows that contain no flux keep their nominal reference). Returns a NEW
    ``SubApertureGrid`` (the input is not mutated). The grid's ordering / window
    size are preserved, so the calibrated references stay in canonical order.
    """
    if grid is None:
        grid = build_subaperture_grid(cfg)

    frame = np.asarray(flat_frame, dtype=np.float64)
    H, W = frame.shape
    win = grid.w
    n = grid.ref_x.shape[0]

    new_ref_x = grid.ref_x.astype(np.float64).copy()
    new_ref_y = grid.ref_y.astype(np.float64).copy()

    for k in range(n):
        # Integer window aligned to the nominal top-left, clipped to frame.
        xi0 = int(np.floor(grid.x0[k]))
        yi0 = int(np.floor(grid.y0[k]))
        xi1 = xi0 + win
        yi1 = yi0 + win
        cx0 = max(xi0, 0)
        cy0 = max(yi0, 0)
        cx1 = min(xi1, W)
        cy1 = min(yi1, H)
        if cx1 <= cx0 or cy1 <= cy0:
            continue
        sub = frame[cy0:cy1, cx0:cx1]
        tot = float(sub.sum())
        if tot <= 0.0:
            continue
        ys = np.arange(cy0, cy1)
        xs = np.arange(cx0, cx1)
        gx = (sub.sum(axis=0) * xs).sum() / tot
        gy = (sub.sum(axis=1) * ys).sum() / tot
        new_ref_x[k] = gx
        new_ref_y[k] = gy

    return SubApertureGrid(
        x0=grid.x0.copy(), y0=grid.y0.copy(),
        w=grid.w, h=grid.h,
        ref_x=new_ref_x, ref_y=new_ref_y,
        valid=grid.valid.copy(),
        ij=grid.ij.copy(),
    )


def set_references(grid: SubApertureGrid, ref_x: np.ndarray, ref_y: np.ndarray
                   ) -> SubApertureGrid:
    """Override the reference centroids from externally measured values.

    ``ref_x``/``ref_y`` must be length ``grid.ref_x.shape[0]`` and in the same
    (canonical) order as ``grid``. Returns a NEW grid; the input is unchanged.
    Use this when the flat-wavefront references were computed elsewhere (e.g.
    averaged over many flats, or by the C core).
    """
    ref_x = np.asarray(ref_x, dtype=np.float64)
    ref_y = np.asarray(ref_y, dtype=np.float64)
    if ref_x.shape != grid.ref_x.shape or ref_y.shape != grid.ref_y.shape:
        raise ValueError(
            f"reference arrays must have shape {grid.ref_x.shape}; "
            f"got {ref_x.shape} and {ref_y.shape}"
        )
    return SubApertureGrid(
        x0=grid.x0.copy(), y0=grid.y0.copy(),
        w=grid.w, h=grid.h,
        ref_x=ref_x.copy(), ref_y=ref_y.copy(),
        valid=grid.valid.copy(),
        ij=grid.ij.copy(),
    )


def active_subaperture_mask(cfg: Config, flat_frame: np.ndarray,
                            grid: SubApertureGrid,
                            flux_frac: float = 0.5) -> np.ndarray:
    """Boolean mask of illuminated sub-apertures (flux >= ``flux_frac`` of the
    unobscured-cell flux). research/01 S4.

    The window flux of each cell of ``grid`` is measured on ``flat_frame``; the
    "unobscured" reference is the **maximum** window flux (a fully-illuminated
    interior cell). Cells whose flux is at least ``flux_frac`` of that maximum
    are kept. Returns a bool array aligned with ``grid`` (length
    ``grid.ref_x.shape[0]``).
    """
    frame = np.asarray(flat_frame, dtype=np.float64)
    H, W = frame.shape
    win = grid.w
    n = grid.ref_x.shape[0]

    flux = np.zeros(n, dtype=np.float64)
    for k in range(n):
        xi0 = int(np.floor(grid.x0[k]))
        yi0 = int(np.floor(grid.y0[k]))
        cx0 = max(xi0, 0)
        cy0 = max(yi0, 0)
        cx1 = min(xi0 + win, W)
        cy1 = min(yi0 + win, H)
        if cx1 <= cx0 or cy1 <= cy0:
            continue
        flux[k] = float(frame[cy0:cy1, cx0:cx1].sum())

    fmax = float(flux.max()) if flux.size else 0.0
    if fmax <= 0.0:
        return np.zeros(n, dtype=bool)
    return flux >= float(flux_frac) * fmax


def pupil_mask(cfg: Config, n_grid: int) -> np.ndarray:
    """Circular pupil mask on an ``n_grid`` x ``n_grid`` array (True inside).

    The disk is centered on the array and spans the full grid (diameter =
    ``n_grid``), i.e. a unit-disk mask resampled onto an ``n_grid`` raster. Used
    for full phase-map reconstruction output and Zernike sampling. The mask is
    symmetric under x- and y-reflection by construction.
    """
    if n_grid < 1:
        raise ValueError("n_grid must be >= 1")
    # Disk of diameter ``n_grid`` inscribed in the array: pixel (i, j) is inside
    # iff its center is within radius ``n_grid / 2`` of the array center
    # ``(n_grid - 1) / 2``. Symmetric under x/y reflection; area -> pi/4.
    c = (n_grid - 1) / 2.0
    idx = np.arange(n_grid) - c
    xx, yy = np.meshgrid(idx, idx)
    r = n_grid / 2.0
    return (xx * xx + yy * yy) <= r * r + 1e-9


def build_actuator_grid(cfg: Config,
                        subaps: Optional[SubApertureGrid] = None) -> ActuatorGrid:
    """Fried (n+1)x(n+1) actuator-corner grid aligned to the lenslet grid.
    research/02 S12; research/05 S5.

    Actuator nodes sit at the **corners** of the lenslet sub-apertures, row-major
    (``flat = row * (n_x+1) + col``). Coordinates are returned both in detector
    pixels (``x, y``) and as normalized unit-disk coordinates (``x_norm,
    y_norm``, divided by the pupil radius) for evaluating the Zernike
    basis/gradients. The ``valid`` mask flags actuators that touch at least one
    valid sub-aperture; if ``subaps`` is omitted a nominal grid is built to
    determine validity.
    """
    ax, ay, ij = _corner_nodes_px(cfg)
    R = pupil_radius_px(cfg)
    cx0, cy0 = cfg.pupil.center_x_px, cfg.pupil.center_y_px
    x_norm = (ax - cx0) / R
    y_norm = (ay - cy0) / R

    if subaps is None:
        subaps = build_subaperture_grid(cfg, valid_only=True)

    n_act = ax.shape[0]
    valid = np.zeros(n_act, dtype=bool)
    corner_idx = fried_corner_indices(cfg.mla.n_lenslets_x, cfg.mla.n_lenslets_y)
    # corner_idx is over ALL lenslets (row-major); pick the rows for valid subaps.
    nx = cfg.mla.n_lenslets_x
    for col, row in subaps.ij:
        flat_lens = int(row) * nx + int(col)
        for a in corner_idx[flat_lens]:
            valid[int(a)] = True

    return ActuatorGrid(
        x=ax, y=ay, ij=ij,
        x_norm=x_norm, y_norm=y_norm,
        valid=valid,
    )


def fried_corner_indices(n_lenslets_x: int, n_lenslets_y: int) -> np.ndarray:
    """For each sub-aperture, the (a,b,c,d)=(TL,TR,BL,BR) corner-phase indices
    on the (n+1)x(n+1) grid, row-major. Used to build the Fried Gamma matrix.
    research/02 S4.

    Returns an ``(n_x*n_y, 4)`` int array indexed by lenslet flat index
    ``row * n_x + col`` (the canonical lenslet order). Corner indices are on the
    row-major (n_x+1)x(n_y+1) corner grid:
        a (TL) = row*(n_x+1) + col
        b (TR) = row*(n_x+1) + col + 1
        c (BL) = (row+1)*(n_x+1) + col
        d (BR) = (row+1)*(n_x+1) + col + 1
    """
    nx, ny = int(n_lenslets_x), int(n_lenslets_y)
    stride = nx + 1
    cols = np.arange(nx)
    rows = np.arange(ny)
    cc, rr = np.meshgrid(cols, rows)        # row-major (rr slowest)
    cc = cc.ravel()
    rr = rr.ravel()
    a = rr * stride + cc
    b = a + 1
    c = (rr + 1) * stride + cc
    d = c + 1
    return np.stack([a, b, c, d], axis=1).astype(np.int64)


def lenslet_actuator_maps(grid: SubApertureGrid, acts: ActuatorGrid
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """Index maps: for each lenslet, the 4 surrounding actuator-corner indices
    (and the inverse). Fried adjacency. research/02 S12, research/05 S5.

    ``forward`` is ``(M, 4)`` -- for each *valid* sub-aperture (in canonical
    order), its ``(a, b, c, d)=(TL,TR,BL,BR)`` corner-actuator indices on the
    row-major corner grid. ``inverse`` is ``(n_act, 4)`` -- for each actuator,
    the up-to-4 valid-sub-aperture indices that share it (``-1`` padding), in
    increasing sub-aperture order. Together they encode the Fried lenslet<->
    actuator adjacency consumed by the DM mapping.
    """
    # Infer grid width from the actuator (col,row) indices: stride = n_x + 1.
    nx1 = int(acts.ij[:, 0].max()) + 1          # n_lenslets_x + 1
    nx = nx1 - 1
    stride = nx1

    M = grid.ij.shape[0]
    forward = np.empty((M, 4), dtype=np.int64)
    for k in range(M):
        col, row = int(grid.ij[k, 0]), int(grid.ij[k, 1])
        a = row * stride + col
        forward[k] = (a, a + 1, a + stride, a + stride + 1)

    n_act = acts.x.shape[0]
    inverse = np.full((n_act, 4), -1, dtype=np.int64)
    fill = np.zeros(n_act, dtype=np.int64)
    for k in range(M):
        for a in forward[k]:
            ai = int(a)
            if fill[ai] < 4:
                inverse[ai, fill[ai]] = k
                fill[ai] += 1
    return forward, inverse


def slope_vector_layout(n_sub: int) -> dict:
    """Document the canonical 2M slope-vector layout (the cross-module contract).

    Returns a dict describing the **block** layout
    ``s = [sx_1 .. sx_M, sy_1 .. sy_M]`` (ARCHITECTURE.md S2): indices
    ``0 .. M-1`` are x-slopes and ``M .. 2M-1`` are y-slopes, where each
    sub-aperture appears in canonical lenslet order (see module docstring). This
    is the single source of truth shared by ``slopes_from_centroids`` and the C
    real-time core.
    """
    M = int(n_sub)
    return {
        "layout": "block",
        "order": "[sx_1..sx_M, sy_1..sy_M]",
        "n_sub": M,
        "n_slopes": 2 * M,
        "x_slice": (0, M),
        "y_slice": (M, 2 * M),
        "lenslet_order": "row-major valid sub-apertures (row*n_lenslets_x + col)",
    }


def slopes_from_centroids(cfg: Config,
                          cx: np.ndarray, cy: np.ndarray,
                          ref_x: np.ndarray, ref_y: np.ndarray) -> np.ndarray:
    """Convert measured spot centroids (px) to a 2M slope vector (radians).

    ``slope = (centroid - reference) * pixel_size_m / focal_length_m``
    (research/01 S1.1, research/02 S1; only ``p_pix`` and ``f_MLA`` enter). The
    output uses the canonical **block** layout ``[sx_1..sx_M, sy_1..sy_M]``
    (see :func:`slope_vector_layout`). All inputs are length M in canonical
    lenslet order.
    """
    cx = np.asarray(cx, dtype=np.float64)
    cy = np.asarray(cy, dtype=np.float64)
    ref_x = np.asarray(ref_x, dtype=np.float64)
    ref_y = np.asarray(ref_y, dtype=np.float64)
    scale = cfg.slope_scale          # pixel_size_m / focal_length_m
    sx = (cx - ref_x) * scale
    sy = (cy - ref_y) * scale
    return np.concatenate([sx, sy])


def displacement_to_slope(cfg: Config, displacement_px: np.ndarray) -> np.ndarray:
    """Scalar/array helper: pixel displacement -> slope (rad).

    ``slope = displacement_px * pixel_size_m / focal_length_m``. Exact one-liner
    used by tests and callers that already have (centroid - reference).
    """
    return np.asarray(displacement_px, dtype=np.float64) * cfg.slope_scale


def subaperture_centers_norm(cfg: Config, grid: SubApertureGrid
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """Normalized unit-disk (x, y) of the sub-aperture centers (for Zernike).

    ``(ref - pupil_center) / pupil_radius``; both arrays length
    ``grid.ref_x.shape[0]`` in canonical order.
    """
    R = pupil_radius_px(cfg)
    cx0, cy0 = cfg.pupil.center_x_px, cfg.pupil.center_y_px
    return (grid.ref_x - cx0) / R, (grid.ref_y - cy0) / R


def build_geometry(cfg: Config, flux_frac: float = 0.5,
                   flat_frame: Optional[np.ndarray] = None) -> Geometry:
    """Build the full Fried :class:`Geometry` bundle for a configuration.

    Steps (research/02 S1, S5; ARCHITECTURE.md S3.2):
      1. nominal sub-aperture grid + pupil-fill validity (``flux_frac``);
      2. optional flat-frame reference registration + flux-based validity;
      3. (n+1)x(n+1) Fried actuator-corner grid + valid-actuator mask;
      4. lenslet<->actuator adjacency maps;
      5. normalized unit-disk coordinates of sub-aperture centers and actuators.

    If ``flat_frame`` is given, references and the active mask are refined from
    it before selecting valid sub-apertures; otherwise pure pupil geometry is
    used. The returned ``subaps`` contains only valid sub-apertures in canonical
    order (so its arrays index the slope vector directly).
    """
    # Start from the full (all-cells) grid so we can refine validity, then trim.
    full = build_subaperture_grid(cfg, flux_frac=flux_frac, valid_only=False)
    valid = full.valid.copy()

    if flat_frame is not None:
        full = register_references(cfg, flat_frame, full)
        valid = valid & active_subaperture_mask(cfg, flat_frame, full, flux_frac)

    sel = np.where(valid)[0]
    subaps = SubApertureGrid(
        x0=full.x0[sel].copy(), y0=full.y0[sel].copy(),
        w=full.w, h=full.h,
        ref_x=full.ref_x[sel].copy(), ref_y=full.ref_y[sel].copy(),
        valid=np.ones(sel.shape[0], dtype=bool),
        ij=full.ij[sel].copy(),
    )

    acts = build_actuator_grid(cfg, subaps=subaps)
    forward, inverse = lenslet_actuator_maps(subaps, acts)
    sx_norm, sy_norm = subaperture_centers_norm(cfg, subaps)

    return Geometry(
        cfg=cfg,
        subaps=subaps,
        acts=acts,
        corner_idx=forward,
        lenslet_to_act=forward,
        act_to_lenslet=inverse,
        subap_x_norm=sx_norm,
        subap_y_norm=sy_norm,
        pupil_radius_px=pupil_radius_px(cfg),
    )
