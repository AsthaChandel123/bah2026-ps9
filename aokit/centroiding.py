"""aokit.centroiding -- reference centroiders for validation.

Python references for CoG / TCoG / WCoG / TWCoG / IWCoG / brightest-pixel
/ correlation / Gaussian-fit, used to validate the C TWCoG kernel and to serve
as ground-truth oracles (Gaussian-fit / ML are offline references). research/01.

All single-window estimators operate on a 2-D sub-aperture window ``I[row, col]``
and return a sub-pixel ``(cx, cy)`` in **window-local pixel coordinates**, where
``cx`` is the column (x / fast) axis and ``cy`` the row (y / slow) axis:

    cx = Sum_{r,c} c * I[r, c] / Sum I        (x, column)
    cy = Sum_{r,c} r * I[r, c] / Sum I        (y, row)

This (x=col, y=row) convention is consistent with the spot-displacement model in
research/01 S1.1 and with ``aokit.geometry`` reference centroids (``ref_x`` is a
column position, ``ref_y`` a row position).

Robustness: every estimator returns ``(nan, nan)`` for a window with no usable
flux (all-zero after thresholding) rather than raising, so a single dead
sub-aperture cannot crash the frame loop. The driver replaces such NaNs with the
reference centroid (i.e. zero slope) by default.

The slope ordering produced by :func:`centroids_to_slopes` matches
ARCHITECTURE.md S3.2 / S4.2: ``s = [sx_1..sx_M, sy_1..sy_M]`` (all x-slopes,
then all y-slopes), over the *valid* sub-apertures in grid order.
"""
from __future__ import annotations

from typing import Tuple, Optional
import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_NAN2 = (float("nan"), float("nan"))


def _as_f64(window: np.ndarray) -> np.ndarray:
    """Return ``window`` as a contiguous 2-D float64 array (no copy if able)."""
    arr = np.asarray(window, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"centroid window must be 2-D, got shape {arr.shape}")
    return arr


def _moments(weighted: np.ndarray) -> Tuple[float, float]:
    """First-moment centroid of a non-negative ``weighted`` array.

    ``cx`` is the column-index moment, ``cy`` the row-index moment. Returns
    ``(nan, nan)`` if the total is non-positive (empty / dead window).
    """
    total = float(weighted.sum())
    if not np.isfinite(total) or total <= 0.0:
        return _NAN2
    h, w = weighted.shape
    cols = np.arange(w, dtype=np.float64)
    rows = np.arange(h, dtype=np.float64)
    cx = float(weighted.sum(axis=0) @ cols) / total
    cy = float(weighted.sum(axis=1) @ rows) / total
    return cx, cy


# ---------------------------------------------------------------------------
# M1 -- plain CoG
# ---------------------------------------------------------------------------

def cog(window: np.ndarray) -> Tuple[float, float]:
    """Plain center-of-gravity (first moment). research/01 M1.

    Returns window-local ``(cx, cy)``. Negative pixel values are honored as-is
    (the caller is expected to have removed bias/background); use
    :func:`thresholded_cog` to clip a noise floor first.
    """
    arr = _as_f64(window)
    return _moments(arr)


# Backwards/forwards-friendly alias matching the task brief's naming.
center_of_gravity = cog


# ---------------------------------------------------------------------------
# M2 -- thresholded CoG
# ---------------------------------------------------------------------------

def thresholded_cog(window: np.ndarray, thresh: Optional[float] = None,
                    frac: Optional[float] = None) -> Tuple[float, float]:
    """Thresholded CoG: subtract a threshold, clip to 0, then CoG. research/01 M2.

    Parameters
    ----------
    window : 2-D array
        Sub-aperture image ``I[row, col]``.
    thresh : float, optional
        Absolute threshold level ``I_T`` (e.g. ``m * sigma_read`` or a measured
        background). Subtracted from every pixel before clipping to zero.
    frac : float, optional
        Fraction-of-max threshold; if given (and ``thresh`` is None), uses
        ``I_T = frac * I_max``. ``research/01`` recommends ``frac ~ 0.05-0.10``.

    Exactly one of ``thresh`` / ``frac`` is used. If both are None, this reduces
    to plain :func:`cog`. If both are given, the larger effective level is used
    (``I_T = max(thresh, frac * I_max)``), mirroring the TWCoG noise-floor rule
    ``I_T = max(T*I_max, m*sigma_read)``.
    """
    arr = _as_f64(window)
    level = 0.0
    if thresh is not None:
        level = max(level, float(thresh))
    if frac is not None:
        imax = float(arr.max()) if arr.size else 0.0
        level = max(level, float(frac) * imax)
    if level > 0.0:
        arr = np.clip(arr - level, 0.0, None)
    return _moments(arr)


# ---------------------------------------------------------------------------
# M3 -- weighted CoG
# ---------------------------------------------------------------------------

def weighted_cog(window: np.ndarray,
                 weights: Optional[np.ndarray] = None,
                 sigma: Optional[float] = None) -> Tuple[float, float]:
    """Weighted CoG with a (precomputed or Gaussian) weight. research/01 M3.

    Parameters
    ----------
    window : 2-D array
        Sub-aperture image.
    weights : 2-D array, optional
        Precomputed weight window ``W[row, col]`` (same shape as ``window``).
        This is the real-time form (a weight LUT, no runtime ``exp``).
    sigma : float, optional
        If ``weights`` is None, build a Gaussian weight of this 1-sigma width
        (pixels) centered on the window center. (Use :func:`twcog` for a weight
        re-centered on the spot, which removes most of the WCoG center-pull
        bias.)

    Notes
    -----
    Fixed-center Gaussian weighting is the maximum-likelihood spot-position
    estimator under Gaussian noise *when the weight matches the spot and is
    centered on it*; an off-center fixed weight introduces a known center-pull
    bias (research/01 M3, S8) which reference subtraction cancels common-mode.
    """
    arr = _as_f64(window)
    if weights is None:
        if sigma is None or sigma <= 0:
            raise ValueError("weighted_cog needs either `weights` or sigma>0")
        h, w = arr.shape
        cy0 = (h - 1) / 2.0
        cx0 = (w - 1) / 2.0
        weights = gaussian_weight(w, h, cx0, cy0, fwhm_px=2.3548200450309493 * sigma)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != arr.shape:
            raise ValueError(
                f"weights shape {weights.shape} != window shape {arr.shape}")
    return _moments(arr * weights)


# ---------------------------------------------------------------------------
# Gaussian weight LUT
# ---------------------------------------------------------------------------

def gaussian_weight(win_w: int, win_h: int, cx: float, cy: float,
                    fwhm_px: float) -> np.ndarray:
    """Precompute a Gaussian weight window centred at ``(cx, cy)`` for WCoG/TWCoG.

    ``(cx, cy)`` are window-local (col, row) coordinates. ``fwhm_px`` is the
    full-width-at-half-maximum of the weight (research/01 recommends a weight
    FWHM ~ spot FWHM ~ 4-4.5 px). Returns a ``(win_h, win_w)`` float64 array
    normalized to a peak of 1.0.
    """
    if win_w <= 0 or win_h <= 0:
        raise ValueError("window dimensions must be positive")
    if fwhm_px <= 0:
        raise ValueError("fwhm_px must be positive")
    sigma = float(fwhm_px) / 2.3548200450309493  # FWHM -> sigma
    cols = np.arange(win_w, dtype=np.float64)[None, :]
    rows = np.arange(win_h, dtype=np.float64)[:, None]
    r2 = (cols - cx) ** 2 + (rows - cy) ** 2
    return np.exp(-r2 / (2.0 * sigma * sigma))


# ---------------------------------------------------------------------------
# S12 -- TWCoG (PRIMARY)
# ---------------------------------------------------------------------------

def twcog(window: np.ndarray, weights: Optional[np.ndarray] = None,
          thresh_frac: float = 0.05, thresh_sigma: float = 0.0,
          gain: float = 1.0, min_pixels: int = 3,
          fwhm_px: float = 4.0) -> Tuple[float, float]:
    """Thresholded + Windowed Weighted CoG (the PRIMARY estimator; the C core
    mirrors this). research/01 S12.

    Steps (per research/01 S12):

    1. Threshold at ``I_T = max(thresh_frac * I_max, thresh_sigma)`` and clip
       negatives. ``thresh_sigma`` is an absolute level (``m * sigma_read`` in
       physical use); ``thresh_frac`` a fraction of the window max.
    2. Validity gate: require at least ``min_pixels`` pixels strictly above the
       threshold, else return ``(nan, nan)``.
    3. Weighted first moments with a Gaussian weight **centered on the spot**
       (the thresholded-CoG estimate), FWHM ``fwhm_px`` -- a single windowing
       pass that removes the fixed-weight center-pull bias. If ``weights`` is
       supplied (a precomputed LUT, as the C core uses), it is used verbatim and
       no re-centering is done.
    4. Apply the precomputed WCoG ``gain`` about the window center (linearizes
       residual weight pull; ``gain=1`` is a no-op).

    Returns window-local ``(cx, cy)`` or ``(nan, nan)`` for a dead/under-filled
    window.
    """
    arr = _as_f64(window)
    if arr.size == 0:
        return _NAN2

    imax = float(arr.max())
    level = max(float(thresh_frac) * imax, float(thresh_sigma), 0.0)
    if level > 0.0:
        thr = np.clip(arr - level, 0.0, None)
    else:
        thr = arr.copy()

    # Validity gate: >= min_pixels above threshold.
    n_above = int(np.count_nonzero(thr > 0.0)) if level > 0.0 \
        else int(np.count_nonzero(arr > 0.0))
    if n_above < int(min_pixels):
        return _NAN2

    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != arr.shape:
            raise ValueError(
                f"weights shape {w.shape} != window shape {arr.shape}")
        cx, cy = _moments(thr * w)
    else:
        # Window the weight on the spot location (thresholded CoG seed).
        cx0, cy0 = _moments(thr)
        if not (np.isfinite(cx0) and np.isfinite(cy0)):
            return _NAN2
        h, ww = arr.shape
        w = gaussian_weight(ww, h, cx0, cy0, fwhm_px=fwhm_px)
        cx, cy = _moments(thr * w)

    if not (np.isfinite(cx) and np.isfinite(cy)):
        return _NAN2

    if gain != 1.0:
        h, ww = arr.shape
        cx_c = (ww - 1) / 2.0
        cy_c = (h - 1) / 2.0
        cx = cx_c + gain * (cx - cx_c)
        cy = cy_c + gain * (cy - cy_c)
    return cx, cy


# ---------------------------------------------------------------------------
# M4 -- iteratively-weighted CoG
# ---------------------------------------------------------------------------

def iter_weighted_cog(window: np.ndarray, sigma_w: float,
                      n_iter: int = 2) -> Tuple[float, float]:
    """Iteratively-weighted CoG (weight re-centred each iter). research/01 M4.

    Seeds the weight center at the plain-CoG estimate, then re-centers a
    Gaussian weight of 1-sigma ``sigma_w`` on the previous estimate for
    ``n_iter`` iterations. A fixed small ``n_iter`` (default 2) keeps the cost
    bounded; this removes most of the fixed-weight WCoG center-pull bias for
    large excursions / low SNR.
    """
    arr = _as_f64(window)
    if sigma_w <= 0:
        raise ValueError("sigma_w must be positive")
    cx, cy = _moments(arr)
    if not (np.isfinite(cx) and np.isfinite(cy)):
        return _NAN2
    h, w = arr.shape
    fwhm = 2.3548200450309493 * float(sigma_w)
    for _ in range(int(n_iter)):
        wts = gaussian_weight(w, h, cx, cy, fwhm_px=fwhm)
        ncx, ncy = _moments(arr * wts)
        if not (np.isfinite(ncx) and np.isfinite(ncy)):
            break
        cx, cy = ncx, ncy
    return cx, cy


# ---------------------------------------------------------------------------
# M7 -- brightest-pixel (Basden)
# ---------------------------------------------------------------------------

def brightest_pixel(window: np.ndarray, n_bright: int) -> Tuple[float, float]:
    """Brightest-pixel selection then CoG (Basden). research/01 M7.

    Keep the ``n_bright`` brightest pixels, subtract the ``n_bright``-th
    brightest value, clip negatives, then CoG -- a data-adaptive threshold that
    tracks per-frame brightness.
    """
    arr = _as_f64(window)
    n = int(n_bright)
    if n <= 0:
        return _NAN2
    flat = arr.ravel()
    if n >= flat.size:
        return _moments(np.clip(arr - float(flat.min()), 0.0, None))
    # value of the n-th brightest pixel (threshold).
    thr = float(np.partition(flat, flat.size - n)[flat.size - n])
    return _moments(np.clip(arr - thr, 0.0, None))


# ---------------------------------------------------------------------------
# M10 -- correlation / matched-filter centroid
# ---------------------------------------------------------------------------

def correlation_centroid(window: np.ndarray, template: np.ndarray
                         ) -> Tuple[float, float]:
    """Correlation/matched-filter centroid (extended/elongated spots).
    research/01 M10. Includes a sub-pixel peak step.

    Cross-correlates ``window`` with ``template`` (same shape) via FFT, finds
    the integer-pixel correlation peak, then refines to sub-pixel with a
    separable 3-point parabolic interpolation per axis (the standard correlation
    sub-pixel refinement, research/01 M8/M10).

    Returns window-local ``(cx, cy)`` of the spot, computed as the template
    centre position shifted by the measured lag, so a template == spot gives
    back the spot centre.

    Robust to uncorrelated noise outside the spot core (correlation suppresses
    it) and to truncation/asymmetry that biases CoG: because the *template*
    encodes the full (possibly elongated) spot shape, the matched-filter peak
    locates the true centre even when part of the spot is clipped by the window
    -- the documented fallback for extended/elongated/structured spots, where
    CoG-family bias explodes. For compact symmetric spots plain CoG is already
    near-optimal, so correlation offers no accuracy gain there.
    """
    arr = _as_f64(window)
    tmpl = _as_f64(template)
    if tmpl.shape != arr.shape:
        raise ValueError(
            f"template shape {tmpl.shape} != window shape {arr.shape}")

    h, w = arr.shape
    # Background-subtract (non-negative) so the matched filter weighs the spot,
    # not a DC pedestal; keep the peak shape undistorted (do NOT zero-mean).
    a = np.clip(arr - float(np.median(arr)), 0.0, None)
    t = np.clip(tmpl - float(np.median(tmpl)), 0.0, None)

    # Full linear cross-correlation via FFT (circular-correlation of zero-padded
    # inputs == linear correlation). Pad to size 2N-1 for full overlap.
    ph = 2 * h - 1
    pw = 2 * w - 1
    fa = np.fft.rfft2(a, s=(ph, pw))
    ft = np.fft.rfft2(t, s=(ph, pw))
    corr = np.fft.irfft2(fa * np.conj(ft), s=(ph, pw))
    # Shift so zero-lag is centered; lag index runs -(N-1) .. +(N-1).
    corr = np.fft.fftshift(corr)
    center_r = ph // 2
    center_c = pw // 2

    pr, pc = np.unravel_index(int(np.argmax(corr)), corr.shape)

    def _parab(cm1: float, c0: float, cp1: float) -> float:
        """Sub-sample offset in [-1, 1] from a 3-point parabola peak fit."""
        denom = (cm1 - 2.0 * c0 + cp1)
        if denom == 0.0:
            return 0.0
        d = 0.5 * (cm1 - cp1) / denom
        if not np.isfinite(d) or abs(d) > 1.0:
            return 0.0
        return float(d)

    dr = 0.0
    if 0 < pr < corr.shape[0] - 1:
        dr = _parab(float(corr[pr - 1, pc]), float(corr[pr, pc]),
                    float(corr[pr + 1, pc]))
    dc = 0.0
    if 0 < pc < corr.shape[1] - 1:
        dc = _parab(float(corr[pr, pc - 1]), float(corr[pr, pc]),
                    float(corr[pr, pc + 1]))

    lag_r = (pr + dr) - center_r   # row shift of window vs template
    lag_c = (pc + dc) - center_c   # col shift

    # Template centre-of-mass in its own window; the spot in `window` sits at
    # template-centre + lag.
    tcx, tcy = _moments(np.clip(tmpl - float(tmpl.min()), 0.0, None))
    if not (np.isfinite(tcx) and np.isfinite(tcy)):
        tcx = (w - 1) / 2.0
        tcy = (h - 1) / 2.0
    return tcx + lag_c, tcy + lag_r


# ---------------------------------------------------------------------------
# M9 -- 2-D Gaussian least-squares fit (offline ground truth)
# ---------------------------------------------------------------------------

def gaussfit_centroid(window: np.ndarray) -> Tuple[float, float]:
    """2-D Gaussian least-squares fit centroid (offline ground truth).
    research/01 M9.

    Fits ``I(c,r) ~= B + A * exp(-((c-x0)^2 + (r-y0)^2) / (2 sigma^2))`` by
    nonlinear least squares (``scipy.optimize.curve_fit``, Levenberg-Marquardt),
    seeded from the thresholded CoG and the window's peak/background. Returns
    window-local ``(x0, y0) = (cx, cy)``. Falls back to the thresholded-CoG
    estimate if the optimizer fails to converge.

    This is the high-accuracy offline oracle used to benchmark TWCoG against the
    Cramer-Rao bound; it is *not* used in the real-time loop.
    """
    arr = _as_f64(window)
    h, w = arr.shape
    if arr.size < 5:
        return _moments(arr)

    cols = np.arange(w, dtype=np.float64)
    rows = np.arange(h, dtype=np.float64)
    cc, rr = np.meshgrid(cols, rows)  # cc[r,c]=c, rr[r,c]=r
    cc = cc.ravel()
    rr = rr.ravel()
    z = arr.ravel()

    # Seeds.
    bg0 = float(np.median(arr))
    amp0 = float(arr.max() - bg0)
    if amp0 <= 0:
        amp0 = float(arr.max()) if arr.max() > 0 else 1.0
    cx0, cy0 = thresholded_cog(arr, frac=0.1)
    if not (np.isfinite(cx0) and np.isfinite(cy0)):
        cx0, cy0 = (w - 1) / 2.0, (h - 1) / 2.0
    sigma0 = max(1.0, 0.25 * min(w, h))

    def model(xy, amp, x0, y0, sigma, bg):
        c, r = xy
        s2 = 2.0 * sigma * sigma
        return bg + amp * np.exp(-((c - x0) ** 2 + (r - y0) ** 2) / s2)

    try:
        from scipy.optimize import curve_fit
        p0 = [amp0, cx0, cy0, sigma0, bg0]
        bounds = (
            [0.0, -1.0, -1.0, 0.25, -np.inf],
            [np.inf, w, h, max(w, h), np.inf],
        )
        popt, _ = curve_fit(model, (cc, rr), z, p0=p0, bounds=bounds,
                            maxfev=10000)
        x0f, y0f = float(popt[1]), float(popt[2])
        if np.isfinite(x0f) and np.isfinite(y0f):
            return x0f, y0f
    except Exception:
        pass
    return cx0, cy0


# ---------------------------------------------------------------------------
# Driver: full frame + sub-aperture grid -> centroids -> slopes
# ---------------------------------------------------------------------------

# Registry of single-window methods callable by name from the driver.
_METHODS = {
    "cog": lambda win, **kw: cog(win),
    "tcog": lambda win, **kw: thresholded_cog(
        win, thresh=kw.get("thresh"), frac=kw.get("frac", 0.05)),
    "thresholded_cog": lambda win, **kw: thresholded_cog(
        win, thresh=kw.get("thresh"), frac=kw.get("frac", 0.05)),
    "wcog": lambda win, **kw: weighted_cog(
        win, weights=kw.get("weights"), sigma=kw.get("sigma")),
    "weighted_cog": lambda win, **kw: weighted_cog(
        win, weights=kw.get("weights"), sigma=kw.get("sigma")),
    "twcog": lambda win, **kw: twcog(
        win, weights=kw.get("weights"),
        thresh_frac=kw.get("thresh_frac", 0.05),
        thresh_sigma=kw.get("thresh_sigma", 0.0),
        gain=kw.get("gain", 1.0), min_pixels=kw.get("min_pixels", 3),
        fwhm_px=kw.get("fwhm_px", 4.0)),
    "iwcog": lambda win, **kw: iter_weighted_cog(
        win, sigma_w=kw.get("sigma_w", 2.0), n_iter=kw.get("n_iter", 2)),
    "brightest": lambda win, **kw: brightest_pixel(
        win, n_bright=kw.get("n_bright", 9)),
    "gaussfit": lambda win, **kw: gaussfit_centroid(win),
    "correlation": lambda win, **kw: correlation_centroid(win, kw["template"]),
}


def _extract_window(frame: np.ndarray, x0: float, y0: float,
                    w: int, h: int) -> Tuple[np.ndarray, int, int]:
    """Slice an ``h x w`` window from ``frame`` at integer top-left (x0, y0),
    clamped to the frame bounds. Returns ``(window, ix0, iy0)`` where
    ``(ix0, iy0)`` is the actual clamped top-left used (for coordinate mapping).
    """
    fh, fw = frame.shape[:2]
    ix0 = int(round(x0))
    iy0 = int(round(y0))
    ix0 = max(0, min(ix0, fw - 1))
    iy0 = max(0, min(iy0, fh - 1))
    ix1 = min(fw, ix0 + int(w))
    iy1 = min(fh, iy0 + int(h))
    return frame[iy0:iy1, ix0:ix1], ix0, iy0


def centroid_frame(frame: np.ndarray, grid, method: str = "twcog",
                   only_valid: bool = True, **kwargs) -> np.ndarray:
    """Centroid every sub-aperture of ``frame`` using ``grid`` and ``method``.

    Parameters
    ----------
    frame : 2-D array
        Full detector frame ``I[row, col]`` (already preprocessed: dark/flat/
        background removed by the caller).
    grid : SubApertureGrid (from ``aokit.geometry``)
        Provides per-lenslet window top-left ``(x0, y0)``, window size
        ``(w, h)``, and the ``valid`` mask. (Imported lazily by the caller; this
        function only relies on the duck-typed attributes ``x0, y0, w, h,
        valid``.)
    method : str
        One of the keys in the method registry: ``cog, tcog, wcog, twcog,
        iwcog, brightest, gaussfit, correlation``. Default ``twcog`` (the
        primary estimator).
    only_valid : bool
        If True (default), entries for invalid sub-apertures are ``nan`` and are
        skipped (their windows are not centroided).
    **kwargs :
        Method-specific parameters (e.g. ``thresh_frac``, ``fwhm_px``,
        ``template`` for correlation, ``weights`` for a precomputed LUT).

    Returns
    -------
    centroids : ``(n_sub, 2)`` float64 array
        Absolute (full-frame) centroid ``[cx, cy]`` per sub-aperture, i.e.
        window-local result plus the window's top-left offset. Dead/invalid
        sub-apertures are ``(nan, nan)``.
    """
    if method not in _METHODS:
        raise ValueError(
            f"unknown centroiding method '{method}'; "
            f"choose from {sorted(_METHODS)}")
    fn = _METHODS[method]

    frame = np.asarray(frame, dtype=np.float64)
    x0 = np.asarray(grid.x0)
    y0 = np.asarray(grid.y0)
    w = int(grid.w)
    h = int(grid.h)
    valid = np.asarray(grid.valid, dtype=bool)
    n_sub = x0.shape[0]

    out = np.full((n_sub, 2), np.nan, dtype=np.float64)
    for k in range(n_sub):
        if only_valid and not valid[k]:
            continue
        win, ix0, iy0 = _extract_window(frame, float(x0[k]), float(y0[k]), w, h)
        if win.size == 0:
            continue
        cx, cy = fn(win, **kwargs)
        if np.isfinite(cx) and np.isfinite(cy):
            out[k, 0] = cx + ix0
            out[k, 1] = cy + iy0
    return out


def centroids_to_slopes(centroids: np.ndarray, grid, cfg,
                        ref_centroids: Optional[np.ndarray] = None,
                        fill_dead: str = "reference") -> np.ndarray:
    """Convert absolute centroids to the canonical slope vector. research/01 S1.

    ``s = (centroid - reference) * (pixel_size_m / focal_length_m)`` [rad].

    Reference-slope subtraction (research/01 S5, S8): the reference centroids are
    the flat-wavefront spot positions measured with the *same* estimator, so
    common-mode bias (WCoG center-pull, threshold bias, pixel-locking) cancels.

    Parameters
    ----------
    centroids : ``(n_sub, 2)`` array
        Absolute ``[cx, cy]`` per sub-aperture (from :func:`centroid_frame`).
    grid : SubApertureGrid
        Supplies ``ref_x, ref_y`` (reference centroids) and the ``valid`` mask.
    cfg : Config
        Supplies ``slope_scale`` = ``pixel_size_m / focal_length_m``.
    ref_centroids : ``(n_sub, 2)`` array, optional
        Override reference centroids (e.g. measured with the live estimator on
        the flat frame). Defaults to ``(grid.ref_x, grid.ref_y)``.
    fill_dead : {"reference", "nan", "zero"}
        What to emit for a dead (NaN) sub-aperture: ``"reference"`` -> zero
        slope (centroid := reference); ``"zero"`` -> 0.0 explicitly;
        ``"nan"`` -> leave NaN. Default ``"reference"`` so the slope vector has
        no NaNs for the downstream MVM.

    Returns
    -------
    slopes : ``(2 * n_valid,)`` float64 array
        Canonical ordering ``[sx_1..sx_M, sy_1..sy_M]`` over the **valid**
        sub-apertures (ARCHITECTURE.md S3.2 / S4.2). ``M = n_valid``.
    """
    centroids = np.asarray(centroids, dtype=np.float64)
    valid = np.asarray(grid.valid, dtype=bool)

    if ref_centroids is None:
        ref = np.stack([np.asarray(grid.ref_x, dtype=np.float64),
                        np.asarray(grid.ref_y, dtype=np.float64)], axis=1)
    else:
        ref = np.asarray(ref_centroids, dtype=np.float64)

    disp = centroids - ref  # (n_sub, 2) in px

    # Handle dead sub-apertures (NaN centroid).
    dead = ~np.isfinite(disp).all(axis=1)
    if fill_dead in ("reference", "zero"):
        disp[dead, :] = 0.0
    elif fill_dead == "nan":
        pass
    else:
        raise ValueError("fill_dead must be 'reference', 'zero', or 'nan'")

    scale = float(cfg.slope_scale)
    disp_valid = disp[valid, :] * scale  # (M, 2) rad

    # Canonical vector: all x-slopes then all y-slopes.
    return np.concatenate([disp_valid[:, 0], disp_valid[:, 1]])


def frame_to_slopes(frame: np.ndarray, grid, cfg, method: str = "twcog",
                    ref_centroids: Optional[np.ndarray] = None,
                    fill_dead: str = "reference", **kwargs) -> np.ndarray:
    """End-to-end convenience: frame -> centroids -> reference-subtracted slopes.

    Equivalent to ``centroid_frame`` followed by ``centroids_to_slopes``;
    returns the canonical ``[sx_1..sx_M, sy_1..sy_M]`` slope vector over valid
    sub-apertures. See those functions for parameters.
    """
    cents = centroid_frame(frame, grid, method=method, **kwargs)
    return centroids_to_slopes(cents, grid, cfg, ref_centroids=ref_centroids,
                               fill_dead=fill_dead)
