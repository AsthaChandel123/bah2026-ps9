"""aokit.reconstructor -- build zonal + modal reconstructors (offline).

Builds, from a :class:`aokit.geometry.Geometry`:

  - the **Fried gradient (interaction) matrix** ``Gamma`` and the regularized
    **zonal reconstructor** ``R`` (SVD/Tikhonov, piston + waffle removed)
                                                    research/02 S3, S4, S6, S15
  - the **modal interaction matrix** ``M``, the modal reconstructor ``Mpinv``
    (SVD/Tikhonov) and the Zernike **phase-from-coefficients** basis ``Z``
                                                    research/03 S2-S4, S6
  - an **FFT Fourier Transform Reconstructor** (Fried filter) as a cross-check /
    scaling fallback                                research/02 S7

These matrices are serialized to AOMX for the C core (scripts/build_calibration).

Matrix conventions (match the C real-time core; ARCHITECTURE.md S2, S3)
----------------------------------------------------------------------
* **Slope-vector layout** is the canonical BLOCK layout
  ``s = [sx_1 .. sx_M, sy_1 .. sy_M]`` over the ``M`` valid sub-apertures in
  canonical lenslet (row-major) order (see :func:`aokit.geometry.slope_vector_layout`).
* **Phase / actuator unknowns** ``phi`` live at the **valid Fried corner nodes**
  -- the actuator nodes that touch at least one valid sub-aperture
  (``geom.acts.valid``).  ``N_phase`` is that count; nodes are kept in the
  geometry's row-major corner order, compacted to ``0 .. N_phase-1`` by
  :func:`valid_node_index`.
* Shapes: ``Gamma`` is ``(2M, N_phase)``; ``R`` is ``(N_phase, 2M)`` so
  ``phi = R @ s``.  ``M`` (modal) is ``(2M, J)``; ``Mpinv`` is ``(J, 2M)`` so
  ``a = Mpinv @ s``; ``Z`` is ``(N_phase, J)`` so ``W = Z @ a``.

Units of the Fried node spacing ``h``
-------------------------------------
The Fried finite-difference uses a lenslet pitch ``h`` (research/02 S2.2).  Here
the node grid spacing is taken as **unity** (``h = 1``), i.e. the reconstructed
phase ``phi`` is expressed in the natural "phase per node-spacing" units implied
by the slope scaling.  This is the standard zonal/FTR convention and is exactly
what makes the matrix reconstructor and the FFT Fourier reconstructor agree (the
``1/2h`` factor is carried identically in both, see :func:`ftr_reconstruct`).
The round-trip ``s = Gamma @ phi`` -> ``phi_hat = R @ s`` is independent of the
chosen ``h`` (it cancels), so callers that prefer metric units can rescale.
"""
from __future__ import annotations

from typing import Optional, Tuple, Dict
import numpy as np

from .config import Config
from .geometry import Geometry, SubApertureGrid
from . import zernike as _zern


# ============================================================================
# Valid-node bookkeeping (full corner grid -> compact valid-node index)
# ============================================================================

def valid_node_index(geom: Geometry) -> Tuple[np.ndarray, np.ndarray]:
    """Map the full ``(n_x+1)x(n_y+1)`` Fried corner grid to the compact set of
    **valid** phase nodes.

    Returns ``(valid_nodes, remap)`` where

      * ``valid_nodes`` -- ``(N_phase,)`` int array of full-grid corner indices
        that touch >= 1 valid sub-aperture (``geom.acts.valid``), in increasing
        (row-major) order.  ``N_phase = valid_nodes.size``.
      * ``remap`` -- ``(n_act,)`` int array mapping a full-grid corner index to
        its compact ``0 .. N_phase-1`` position (``-1`` for invalid nodes).

    The phase-unknown ordering ``phi = [phi_0 .. phi_{N_phase-1}]`` consumed by
    ``Gamma`` / ``R`` / ``Z`` is exactly this compact order.
    """
    valid_nodes = np.where(geom.acts.valid)[0].astype(np.int64)
    remap = -np.ones(geom.n_act, dtype=np.int64)
    remap[valid_nodes] = np.arange(valid_nodes.size, dtype=np.int64)
    return valid_nodes, remap


def piston_vector(geom: Geometry) -> np.ndarray:
    """The piston null vector ``1`` on the valid phase nodes (``(N_phase,)``)."""
    valid_nodes, _ = valid_node_index(geom)
    return np.ones(valid_nodes.size, dtype=np.float64)


def waffle_vector_nodes(geom: Geometry) -> np.ndarray:
    """The checkerboard **waffle** null-mode vector on the valid phase nodes.

    Waffle is ``(-1)^(col+row)`` evaluated at each valid corner node; it is the
    Fried-geometry sensor null mode (research/02 S6) -- every sub-aperture's
    average slope of waffle is zero, so ``Gamma @ waffle == 0``.  Returned length
    ``N_phase`` in the compact valid-node order.
    """
    valid_nodes, _ = valid_node_index(geom)
    ij = geom.acts.ij[valid_nodes]              # (N_phase, 2) -> (col, row)
    return ((-1.0) ** (ij[:, 0] + ij[:, 1])).astype(np.float64)


def waffle_vector(n_act_x: int, n_act_y: int) -> np.ndarray:
    """The checkerboard waffle null-mode vector on the **full** ``n_act_x x
    n_act_y`` corner grid, row-major (research/02 S6).

    (Kept from the module stub contract.  For the valid-node waffle used by the
    reconstructor see :func:`waffle_vector_nodes`.)
    """
    cols = np.arange(int(n_act_x))
    rows = np.arange(int(n_act_y))
    cc, rr = np.meshgrid(cols, rows)            # row-major (rr slowest)
    return ((-1.0) ** (cc.ravel() + rr.ravel())).astype(np.float64)


# ============================================================================
# 1. Zonal (Fried) interaction matrix Gamma
# ============================================================================

def build_fried_interaction(geom: Geometry, h: float = 1.0) -> np.ndarray:
    """Forward Fried gradient (interaction) matrix ``Gamma`` (``2M x N_phase``).

    Implements the Fried corner-averaging slope model over every valid
    sub-aperture (research/02 S2.2, S4): with the four corner phases
    ``(a, b, c, d) = (TL, TR, BL, BR)`` of a lenslet (from ``geom.corner_idx``),

        s_x = (1/2h) [ (phi_b - phi_a) + (phi_d - phi_c) ]
        s_y = (1/2h) [ (phi_c - phi_a) + (phi_d - phi_b) ]

    Rows are laid out BLOCK -- all ``M`` x-slope rows then all ``M`` y-slope rows
    (matching :func:`aokit.geometry.slope_vector_layout`).  Columns are the
    compact valid phase nodes (:func:`valid_node_index`).

    ``h`` is the node spacing (default ``1.0``; see module docstring on units).
    ``Gamma`` is rank-deficient by exactly 2: piston and waffle are in its null
    space (``Gamma @ 1 == 0``, ``Gamma @ waffle == 0``).
    """
    M = geom.n_sub
    valid_nodes, remap = valid_node_index(geom)
    Np = valid_nodes.size
    ci = geom.corner_idx                         # (M, 4) full-grid (a,b,c,d)
    inv2h = 1.0 / (2.0 * float(h))

    Gx = np.zeros((M, Np), dtype=np.float64)
    Gy = np.zeros((M, Np), dtype=np.float64)
    for k in range(M):
        a = int(remap[int(ci[k, 0])])
        b = int(remap[int(ci[k, 1])])
        c = int(remap[int(ci[k, 2])])
        d = int(remap[int(ci[k, 3])])
        # s_x = (1/2h)[(b - a) + (d - c)]
        Gx[k, b] += inv2h
        Gx[k, a] -= inv2h
        Gx[k, d] += inv2h
        Gx[k, c] -= inv2h
        # s_y = (1/2h)[(c - a) + (d - b)]
        Gy[k, c] += inv2h
        Gy[k, a] -= inv2h
        Gy[k, d] += inv2h
        Gy[k, b] -= inv2h
    return np.vstack([Gx, Gy])


# Backwards-compatible alias matching the original stub signature.
def build_fried_gamma(cfg: Config, grid: SubApertureGrid) -> np.ndarray:
    """Fried gradient matrix ``Gamma`` (stub-compatible wrapper).

    Builds a full :class:`Geometry` from ``cfg`` and delegates to
    :func:`build_fried_interaction`.  ``grid`` is accepted for signature
    compatibility; the matrix is built over the geometry's valid sub-apertures /
    valid corner nodes (research/02 S2.2, S4).
    """
    from .geometry import build_geometry
    return build_fried_interaction(build_geometry(cfg))


# ============================================================================
# 2. Zonal reconstructor R (regularized pseudo-inverse, piston + waffle out)
# ============================================================================

def _project_out(R: np.ndarray, vectors) -> np.ndarray:
    """Left-multiply ``R`` by the orthogonal projector that removes ``vectors``
    from the **output** (row) space: ``R <- (I - sum v v^T/||v||^2) R``.

    Guarantees the reconstructed ``phi = R @ s`` has no component along any
    ``v`` (e.g. piston / waffle), regardless of ``s``.
    """
    R = np.array(R, dtype=np.float64, copy=True)
    for v in vectors:
        v = np.asarray(v, dtype=np.float64)
        nv = float(v @ v)
        if nv <= 0.0:
            continue
        v = v / np.sqrt(nv)
        R = R - np.outer(v, v @ R)
    return R


def build_zonal_reconstructor(Gamma: np.ndarray,
                              reg: str = "tikhonov",
                              alpha: float = 1e-3,
                              sv_rel_threshold: float = 1e-6,
                              remove_piston: bool = True,
                              remove_waffle: bool = True,
                              geom: Optional[Geometry] = None,
                              piston: Optional[np.ndarray] = None,
                              waffle: Optional[np.ndarray] = None,
                              ) -> np.ndarray:
    """Regularized least-squares Fried reconstructor ``R`` (``N_phase x 2M``)
    such that ``phi = R @ s`` (research/02 S3, S6, S15).

    Method
    ------
    1. **SVD** ``Gamma = U S V^T``.  The minimum-norm least-squares inverse is
       ``V S^+ U^T``.  Singular values below ``sv_rel_threshold * S.max()`` are
       treated as null (their reciprocals zeroed) -- this is the **SVD
       truncation** that discards the piston and waffle null directions (whose
       singular values are ~0) plus any other numerically-degenerate modes.
    2. ``S^+`` entries for the kept singular values are formed as

         * ``reg='tsvd'``      : ``1/sigma``                 (truncated SVD)
         * ``reg='tikhonov'``  : ``sigma/(sigma^2 + alpha^2)`` (damped LS)

       Tikhonov damping (default) trades a little bias for a large reduction in
       noise amplification of the smaller singular values.
    3. **Explicit null removal**: the piston ``1`` and the Fried **waffle**
       (checkerboard) vectors are projected out of the reconstructor's output
       space (``R <- (I - P_piston - P_waffle) R``), so ``R`` cannot emit any
       piston or waffle content even if ``s`` is contaminated by sensor noise
       (research/02 S6: SVD threshold *and* explicit waffle removal).

    Parameters
    ----------
    Gamma : (2M, N_phase) ndarray
        The Fried interaction matrix from :func:`build_fried_interaction`.
    reg : {'tikhonov', 'tsvd'}
        Regularization of the kept singular values (default ``'tikhonov'``).
    alpha : float
        Tikhonov damping parameter (ignored for ``'tsvd'``).
    sv_rel_threshold : float
        Relative singular-value cutoff for truncation / null detection.
    remove_piston, remove_waffle : bool
        Project the piston / waffle vector out of ``R`` explicitly.
    geom : Geometry, optional
        Used to build the canonical piston / waffle node vectors.  Required for
        ``remove_waffle`` unless ``waffle`` is supplied; if ``geom`` is given,
        piston/waffle default to the geometry's node vectors.
    piston, waffle : (N_phase,) ndarray, optional
        Explicit null vectors (override the geometry-derived ones).
    """
    Gamma = np.asarray(Gamma, dtype=np.float64)
    U, S, Vt = np.linalg.svd(Gamma, full_matrices=False)

    smax = float(S[0]) if S.size else 0.0
    cutoff = sv_rel_threshold * smax
    kept = S > cutoff

    sinv = np.zeros_like(S)
    if reg == "tsvd":
        sinv[kept] = 1.0 / S[kept]
    elif reg == "tikhonov":
        a2 = float(alpha) ** 2
        # Damped inverse on the kept (non-null) singular values only, so the
        # null directions stay exactly zero rather than ~ sigma/alpha^2.
        sinv[kept] = S[kept] / (S[kept] ** 2 + a2)
    else:
        raise ValueError(f"unknown reg={reg!r} (use 'tikhonov' or 'tsvd')")

    R = (Vt.T * sinv) @ U.T                       # (N_phase, 2M)

    # Resolve the null vectors to project out.
    null_vecs = []
    if remove_piston:
        if piston is not None:
            null_vecs.append(np.asarray(piston, dtype=np.float64))
        elif geom is not None:
            null_vecs.append(piston_vector(geom))
        else:
            null_vecs.append(np.ones(R.shape[0], dtype=np.float64))
    if remove_waffle:
        if waffle is not None:
            null_vecs.append(np.asarray(waffle, dtype=np.float64))
        elif geom is not None:
            null_vecs.append(waffle_vector_nodes(geom))
        # else: cannot build the node-waffle pattern without geometry -> skip
        #       (the SVD truncation already nulls it; explicit removal needs geom).

    if null_vecs:
        R = _project_out(R, null_vecs)
    return R


# ============================================================================
# 3. Modal (Zernike) interaction matrix M
# ============================================================================

def build_modal_interaction(geom: Geometry, j_max: int,
                            exclude_piston: bool = True) -> np.ndarray:
    """Modal interaction matrix ``M`` (``2M x J``) from analytic Zernike
    gradients evaluated at the **sub-aperture-center** normalized coordinates
    (research/03 S2).

    Column ``c`` is the SH-WFS slope response to Zernike mode ``j_c``:

        M[(k, x), c] = dZ_{j_c}/dx (xc_k, yc_k)
        M[(k, y), c] = dZ_{j_c}/dy (xc_k, yc_k)

    where ``(xc_k, yc_k)`` are ``geom.subap_x_norm[k], geom.subap_y_norm[k]``
    (sub-aperture centers on the unit disk).  The analytic Cartesian gradients
    come from :func:`aokit.zernike.zernike_gradient_basis`.  Rows are BLOCK
    layout (all x then all y), matching the slope vector.

    ``exclude_piston=True`` (default) drops Noll ``j = 1`` (piston is
    unobservable by an SH-WFS), so the modes are ``j = 2 .. j_max`` and
    ``J = j_max - 1``.  With ``exclude_piston=False`` the piston column (all
    zeros) is included and ``J = j_max``.
    """
    if exclude_piston:
        j_list = list(range(2, int(j_max) + 1))
    else:
        j_list = list(range(1, int(j_max) + 1))

    xc = np.asarray(geom.subap_x_norm, dtype=np.float64)
    yc = np.asarray(geom.subap_y_norm, dtype=np.float64)
    Gx, Gy = _zern.zernike_gradient_basis(j_list, xc, yc)   # each (M, J)
    return np.vstack([Gx, Gy])                              # (2M, J)


# ============================================================================
# 4. Modal reconstructor Mpinv (SVD pseudo-inverse, truncation / Tikhonov)
# ============================================================================

def build_modal_reconstructor(M: np.ndarray,
                              n_modes: Optional[int] = None,
                              reg: str = "tikhonov",
                              alpha: float = 1e-3,
                              sv_rel_threshold: float = 1e-8) -> np.ndarray:
    """Modal reconstructor ``Mpinv`` (``J x 2M``), ``a = Mpinv @ s`` (research/03
    S2.4, Method 4).

    SVD pseudo-inverse of the modal interaction matrix ``M`` with optional mode
    truncation and Tikhonov damping:

      * ``n_modes`` -- if given, keep only the ``n_modes`` largest singular
        values (truncated SVD; the rest are zeroed).
      * singular values below ``sv_rel_threshold * S.max()`` are also zeroed.
      * ``reg='tikhonov'`` damps the kept reciprocals as
        ``sigma/(sigma^2 + alpha^2)``; ``reg='tsvd'`` uses ``1/sigma``.

    The modal matrix is normally well-conditioned (``J << 2M``), so the default
    damping is light; it mainly guards the smallest singular values.
    """
    M = np.asarray(M, dtype=np.float64)
    U, S, Vt = np.linalg.svd(M, full_matrices=False)

    smax = float(S[0]) if S.size else 0.0
    kept = S > sv_rel_threshold * smax
    if n_modes is not None:
        keep_idx = np.zeros_like(kept)
        keep_idx[:int(n_modes)] = True
        kept = kept & keep_idx

    sinv = np.zeros_like(S)
    if reg == "tsvd":
        sinv[kept] = 1.0 / S[kept]
    elif reg == "tikhonov":
        a2 = float(alpha) ** 2
        sinv[kept] = S[kept] / (S[kept] ** 2 + a2)
    else:
        raise ValueError(f"unknown reg={reg!r} (use 'tikhonov' or 'tsvd')")

    return (Vt.T * sinv) @ U.T                    # (J, 2M)


# ============================================================================
# 5. Phase-from-coefficients basis Z (Zernike at valid nodes)
# ============================================================================

def build_phase_basis(geom: Geometry, j_max: int,
                      exclude_piston: bool = True) -> np.ndarray:
    """Zernike synthesis matrix ``Z`` (``N_phase x J``) at the valid phase nodes
    so that ``W = Z @ a`` (research/03 S0, ARCHITECTURE.md S3.4).

    Column ``c`` is Zernike mode ``j_c`` sampled at the **valid corner-node**
    normalized coordinates (``geom.acts.x_norm / y_norm`` restricted to
    ``geom.acts.valid``).  The mode list matches :func:`build_modal_interaction`
    (``j = 2 .. j_max`` when ``exclude_piston``), so ``Mpinv`` and ``Z`` share
    the same coefficient ordering and ``W = Z @ (Mpinv @ s)`` is consistent.
    """
    if exclude_piston:
        j_list = list(range(2, int(j_max) + 1))
    else:
        j_list = list(range(1, int(j_max) + 1))

    valid_nodes, _ = valid_node_index(geom)
    xn = geom.acts.x_norm[valid_nodes]
    yn = geom.acts.y_norm[valid_nodes]
    return _zern.zernike_basis(j_list, xn, yn)    # (N_phase, J)


# Backwards-compatible alias for the original stub name.
def build_synthesis_matrix(cfg: Config, j_max: int, n_grid: int,
                           pupil_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Zernike synthesis matrix ``Z`` on a regular ``n_grid x n_grid`` disk
    (stub-compatible).  ``W = Z @ a`` over the masked pupil pixels.

    Delegates to :func:`aokit.zernike.zernike_array` (transposed to
    ``(N_pix, J)``).  For the node-sampled synthesis used by the calibration
    builder see :func:`build_phase_basis`.
    """
    Zmodes_by_pix = _zern.zernike_array(int(j_max), int(n_grid),
                                        pupil_mask=pupil_mask)   # (J, N_pix)
    return np.ascontiguousarray(Zmodes_by_pix.T)                 # (N_pix, J)


# ============================================================================
# 6. FFT Fourier Transform Reconstructor (Fried filter)
# ============================================================================

def ftr_fried_filters(shape: Tuple[int, int], h: float = 1.0
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Exact Fried FFT filters ``(gx, gy)`` for an ``(ny, nx)`` grid, with the
    Nyquist row/column zeroed (= waffle removal) (research/02 S7).

        gx = (e^{i fy} + 1) (e^{i fx} - 1) / (2h)
        gy = (e^{i fx} + 1) (e^{i fy} - 1) / (2h)

    ``fx, fy`` are the FFT angular frequencies ``2*pi*fftfreq(n)``.  The
    ``(e^{if}+1)`` factor is the two-row Fried average; its zero at the Nyquist
    frequency ``f = pi`` is exactly the waffle null, so ``gx[ny//2, :] = 0`` and
    ``gy[:, nx//2] = 0`` removes the ``0/0`` and the waffle mode.  The ``1/2h``
    factor matches :func:`build_fried_interaction` so the matrix and FFT
    reconstructors agree.
    """
    ny, nx = int(shape[0]), int(shape[1])
    fy = 2.0 * np.pi * np.fft.fftfreq(ny)
    fx = 2.0 * np.pi * np.fft.fftfreq(nx)
    FX, FY = np.meshgrid(fx, fy)
    inv2h = 1.0 / (2.0 * float(h))
    gx = (np.exp(1j * FY) + 1.0) * (np.exp(1j * FX) - 1.0) * inv2h
    gy = (np.exp(1j * FX) + 1.0) * (np.exp(1j * FY) - 1.0) * inv2h
    gx[ny // 2, :] = 0.0          # Nyquist row  -> waffle removal
    gy[:, nx // 2] = 0.0          # Nyquist col  -> waffle removal
    return gx, gy


def ftr_reconstruct(slopes_x: np.ndarray, slopes_y: np.ndarray,
                    geometry_type: str = "fried", h: float = 1.0) -> np.ndarray:
    """Fourier Transform Reconstructor (Fried): FFT the gridded slopes, divide by
    the least-squares Fried filter, inverse-FFT (research/02 S7).

        Phi_hat(k) = [ gx*(k) X(k) + gy*(k) Y(k) ] / [ |gx(k)|^2 + |gy(k)|^2 ]
        phi = real( IFFT(Phi_hat) ) ,   Phi_hat(0,0) := 0   (piston removed)

    where ``X = FFT2(slopes_x)``, ``Y = FFT2(slopes_y)``.  The Nyquist
    rows/cols of the filter are zeroed (``ftr_fried_filters``), which removes the
    waffle mode; the corresponding ``0/0`` modes of ``Phi_hat`` are set to 0.
    Operates on **square gridded** slopes (a fully-illuminated grid); for a
    circular pupil the slopes should be extended into the dark region first
    (Poyneer boundary extension, research/02 S7).

    Returns the reconstructed phase on the same ``(ny, nx)`` grid as the slopes.
    """
    sx = np.asarray(slopes_x, dtype=np.float64)
    sy = np.asarray(slopes_y, dtype=np.float64)
    if sx.shape != sy.shape or sx.ndim != 2:
        raise ValueError("slopes_x and slopes_y must be 2-D arrays of equal shape")
    if geometry_type != "fried":
        raise ValueError(f"ftr_reconstruct only implements 'fried' (got {geometry_type!r})")

    ny, nx = sx.shape
    gx, gy = ftr_fried_filters((ny, nx), h=h)
    X = np.fft.fft2(sx)
    Y = np.fft.fft2(sy)
    den = np.abs(gx) ** 2 + np.abs(gy) ** 2
    num = np.conj(gx) * X + np.conj(gy) * Y
    Phi = np.zeros_like(num)
    nz = den > 0.0
    Phi[nz] = num[nz] / den[nz]
    Phi[0, 0] = 0.0                       # piston
    return np.real(np.fft.ifft2(Phi))


# ============================================================================
# 7. Direct integration baseline (kept from the stub contract)
# ============================================================================

def integrate_slopes(slopes_x: np.ndarray, slopes_y: np.ndarray,
                     h: float = 1.0) -> np.ndarray:
    """Direct path-integration baseline (cumulative trapezoid; research/02 S5).

    Marches the first column up using ``s_y`` then each row across using
    ``s_x`` (a single integration path).  Path-dependent and noise-accumulating
    -- a first-light sanity baseline, not the reconstructor.  ``slopes_x`` /
    ``slopes_y`` are ``(ny, nx)`` gridded slopes; the result is ``(ny, nx)``
    phase with ``phi[0, 0] = 0``.
    """
    sx = np.asarray(slopes_x, dtype=np.float64)
    sy = np.asarray(slopes_y, dtype=np.float64)
    ny, nx = sx.shape
    phi = np.zeros((ny, nx), dtype=np.float64)
    # First column: integrate s_y up the left edge (trapezoid).
    for j in range(1, ny):
        phi[j, 0] = phi[j - 1, 0] + 0.5 * (sy[j - 1, 0] + sy[j, 0]) * h
    # Each row: integrate s_x across (trapezoid).
    for j in range(ny):
        for i in range(1, nx):
            phi[j, i] = phi[j, i - 1] + 0.5 * (sx[j, i - 1] + sx[j, i]) * h
    return phi


# ============================================================================
# 8. One-shot builder for the calibration pipeline
# ============================================================================

def build_all(geom: Geometry, j_max: int,
              zonal_reg: str = "tikhonov",
              zonal_alpha: float = 1e-3,
              modal_reg: str = "tikhonov",
              modal_alpha: float = 1e-3,
              n_modes: Optional[int] = None,
              exclude_piston: bool = True,
              h: float = 1.0,
              **reg) -> Dict[str, np.ndarray]:
    """Build every reconstruction matrix the calibration builder needs and
    return them as a dict (Wave 3 serializes these to AOMX; here we only return
    arrays).

    Returns
    -------
    dict with keys
        ``'Gamma'``  : (2M, N_phase)  Fried interaction matrix
        ``'R'``      : (N_phase, 2M)  zonal reconstructor (piston+waffle out)
        ``'M'``      : (2M, J)        modal interaction matrix
        ``'Mpinv'``  : (J, 2M)        modal reconstructor
        ``'Z'``      : (N_phase, J)   Zernike phase basis at valid nodes

    Convenience overrides: ``zonal_reg/zonal_alpha`` and ``modal_reg/
    modal_alpha`` tune the two pseudo-inverses; ``n_modes`` truncates the modal
    SVD; ``j_max`` / ``exclude_piston`` fix the Zernike mode list (shared by
    ``M``, ``Mpinv`` and ``Z``).  Extra ``**reg`` keys are accepted and ignored
    for forward-compatibility.
    """
    Gamma = build_fried_interaction(geom, h=h)
    R = build_zonal_reconstructor(
        Gamma, reg=zonal_reg, alpha=zonal_alpha,
        remove_piston=True, remove_waffle=True, geom=geom,
    )
    Mmat = build_modal_interaction(geom, j_max, exclude_piston=exclude_piston)
    Mpinv = build_modal_reconstructor(
        Mmat, n_modes=n_modes, reg=modal_reg, alpha=modal_alpha,
    )
    Z = build_phase_basis(geom, j_max, exclude_piston=exclude_piston)
    return {"Gamma": Gamma, "R": R, "M": Mmat, "Mpinv": Mpinv, "Z": Z}
