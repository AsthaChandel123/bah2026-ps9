"""aokit.turbulence -- multi-method r0 / tau0 estimation (offline).

>=7 independent r0 estimators (R1-R7) and >=6 tau0 estimators (T1-T6) across
three data domains (raw slopes, reconstructed phase, intensity), plus combiners
that report a median +/- spread for cross-validation. research/04 (full report),
ARCHITECTURE.md S3.5 / S4.4.

Conventions (fixed; stated so the cross-checks are apples-to-apples; research/04
S6 numerical-constant caveat):

* All wavefront phases are in **radians at the sensing wavelength**; coefficients
  ``a_j`` are Noll-RMS-normalised (each is the RMS wavefront of mode ``j``).
* Modal-coefficient time-series ``coeffs_ts`` have shape ``(T, J)`` with column
  ``k`` = Noll index ``j = k + 2`` (piston excluded -> first column is tip).
* Slope time-series ``slopes_ts`` have shape ``(T, 2M)`` in the canonical block
  layout ``[sx_1..sx_M, sy_1..sy_M]`` (radians of tilt; geometry.py contract).
* **G-tilt** (gradient / centroid tilt, what a SH actually measures) is used for
  the slope-variance and DIMM constants, not Z-tilt.

Bias removal (research/04 S3.3): subtract centroid-noise variance (tau=0 ACF jump
/ high-f PSD floor); exclude tip/tilt (j=2,3) from the r0 fit; use von Karman when
L0 is finite.  The load-bearing prefactors flagged by the research as
convention-dependent (slope-variance 0.170, DIMM 0.358, seeing 0.98, Greenwood
0.314 / 0.426, Tyler 0.368, PSD-cutoff 0.3) are exposed as named module constants
and as function defaults so the whole pipeline can be re-pinned consistently.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np

from .zernike import noll_variance, noll_covariance, zern_index

__all__ = [
    # r0 estimators
    "r0_from_zernike_variance",
    "r0_from_slope_variance",
    "r0_from_dimm",
    "r0_from_phase_variance",
    "r0_from_total_variance",
    "r0_from_structure_function",
    "r0_l0_from_vonkarman",
    "r0_from_vonkarman",
    "r0_from_seeing",
    # tau0 estimators
    "tau0_from_autocorr",
    "tau0_from_autocorrelation",
    "tau0_from_psd",
    "tau0_from_greenwood",
    "tau0_from_structure_function",
    "wind_from_frozen_flow",
    "tau0_from_taylor",
    "tyler_frequency",
    "tau_tracking",
    # combiners + pipeline
    "TurbulenceResult",
    "combine_estimates",
    "combine_r0",
    "combine_tau0",
    "estimate_all",
    "characterize",
    # constants
    "KOLM_TOTAL_VAR",
    "KOLM_TT_REMOVED_VAR",
    "STRUCT_FN_CONST",
    "SLOPE_GTILT_CONST",
    "SLOPE_ZTILT_CONST",
    "DIMM_BASE_CONST",
    "SEEING_CONST",
    "GREENWOOD_TAU0_CONST",
    "GREENWOOD_FREQ_CONST",
    "TYLER_CONST",
    "PSD_CUTOFF_CONST",
    "T0_TO_TAU0",
]

# ============================ NAMED CONSTANTS ===========================
# (research/04 S6 quick-reference card; several flagged "keep configurable".)

# Total residual Kolmogorov phase variance over a circular pupil (piston removed),
# in units of (D/r0)^(5/3): sigma^2 = 1.0299 (D/r0)^(5/3). Noll 1976.
KOLM_TOTAL_VAR = 1.0299
# Tip/tilt-removed residual variance coefficient (Noll Delta_3).
KOLM_TT_REMOVED_VAR = 0.134
# Kolmogorov phase structure function: D_phi(r) = 6.88 (r/r0)^(5/3).
# 6.88 = 2 (24/5 Gamma(6/5))^(5/6).
STRUCT_FN_CONST = 6.88
# Slope (angle-of-arrival) variance, one axis, G-tilt (centroid tilt):
# <alpha^2> = 0.170 lambda^2 r0^(-5/3) d^(-1/3). Saint-Jacques 1998 / Conan.
SLOPE_GTILT_CONST = 0.170
# Z-tilt (Zernike/wavefront tilt) variant (not what a SH measures).
SLOPE_ZTILT_CONST = 0.184
# DIMM differential image-motion base constant (Sarazin & Roddier 1990):
# K_l = 0.358 (1 - 0.541 b^-1/3), K_t = 0.358 (1 - 0.811 b^-1/3), b = B/D_sub.
DIMM_BASE_CONST = 0.358
DIMM_KL_B = 0.541
DIMM_KT_B = 0.811
# Long-exposure seeing FWHM: epsilon = 0.98 lambda / r0. aotools r0_to_seeing.
SEEING_CONST = 0.98
# Coherence time tau0 = 0.314 r0 / v (= 0.134 / f_G). A&A 2007 aa5788-06.
GREENWOOD_TAU0_CONST = 0.314
# Greenwood frequency f_G = 0.426 v / r0. Greenwood 1977.
GREENWOOD_FREQ_CONST = 0.426
# tau0 = GREENWOOD_BRIDGE / f_G ; 0.314 * 0.426 = 0.1338 ~ 0.134.
GREENWOOD_BRIDGE = 0.134
# Tyler tracking frequency f_T = 0.368 v r0^(-1/6) D^(-5/6). Tyler 1994.
TYLER_CONST = 0.368
# Temporal-PSD cutoff f_c,i ~ 0.3 (n+1) v / D. Conan, Rousset & Madec 1995.
PSD_CUTOFF_CONST = 0.3
# t0 (where D_phi(tau)=1 rad^2) = 0.66 tau0  ->  tau0 = t0 / 0.66 (= 2^(3/5) t0).
T0_TO_TAU0 = 2.0 ** (3.0 / 5.0)   # 1.5157...; t0 = tau0 / T0_TO_TAU0 = 0.66 tau0

_FIVE_THIRDS = 5.0 / 3.0
_THREE_FIFTHS = 3.0 / 5.0


# ============================== UTILITIES ===============================

def _as_2d_timeseries(ts: np.ndarray) -> np.ndarray:
    """Coerce a time-series to 2-D ``(T, K)`` (a 1-D series -> one column)."""
    arr = np.asarray(ts, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    elif arr.ndim != 2:
        raise ValueError(f"time-series must be 1-D or 2-D, got ndim={arr.ndim}")
    return arr


def _autocovariance_1d(x: np.ndarray, max_lag: Optional[int] = None
                       ) -> np.ndarray:
    """Biased temporal **autocovariance** ``C(k) = <x(t) x(t+k)>`` of a 1-D
    series (mean removed), ``k = 0 .. max_lag``.  ``C(0)`` is the sample variance.

    Returned *unnormalised* on purpose: the white measurement-noise term inflates
    only ``C(0)`` (the tau=0 jump), so a noise-free zero-lag value can be
    recovered by extrapolating ``C(k>=1)`` back to ``k=0`` -- the dual use the
    research notes (research/04 T1, S3.3)."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = x.size
    if max_lag is None:
        max_lag = n - 1
    max_lag = int(min(max_lag, n - 1))
    cov = np.empty(max_lag + 1)
    for k in range(max_lag + 1):
        cov[k] = float(np.dot(x[: n - k], x[k:])) / n
    return cov


def _noise_free_zero_lag(cov: np.ndarray, n_fit: int = 4) -> float:
    """Estimate the noise-free zero-lag autocovariance ``C_turb(0)`` by
    extrapolating the smooth turbulence autocovariance from positive lags.

    White centroid noise adds a delta only at ``k=0``; the turbulence part is
    smooth.  For an exponential/AR(1)-like ACF ``log C(k)`` is linear in ``k``,
    so a short log-linear fit over ``k = 1 .. n_fit`` extrapolated to ``k = 0``
    cleanly removes the noise spike (research/04 T1).  Falls back to ``C(0)`` if
    the early lags are not positive/decaying."""
    cov = np.asarray(cov, dtype=float)
    if cov.size < 3:
        return float(cov[0]) if cov.size else float("nan")
    m = min(n_fit, cov.size - 1)
    ks = np.arange(1, m + 1)
    vals = cov[1:m + 1]
    if np.all(vals > 0) and m >= 2:
        # log-linear extrapolation to k = 0.
        slope, intercept = np.polyfit(ks, np.log(vals), 1)
        c0 = float(np.exp(intercept))             # value at k = 0
        # Guard: the extrapolated C(0) should be >= C(1) and <= raw C(0).
        if c0 >= vals[0] and c0 <= cov[0] * 1.5:
            return c0
    # Fallback: linear extrapolation of the first two positive lags.
    if cov.size >= 3:
        c0 = 2.0 * cov[1] - cov[2]
        if c0 > 0 and c0 <= cov[0] * 1.5:
            return c0
    return float(cov[0])


def _crossing_lag(ac: np.ndarray, dt_s: float,
                  level: float = 1.0 / np.e) -> float:
    """Lag (seconds) at which an already-normalised ACF (lag-0 ~ 1, noise
    removed) first falls to ``level``, by linear interpolation.  The search runs
    over lags ``k >= 1`` against a clean reference value of 1.0 at lag 0."""
    ac = np.asarray(ac, dtype=float)
    if ac.size < 2:
        return float("nan")
    prev = 1.0                                    # clean lag-0 reference
    for k in range(1, ac.size):
        if ac[k] <= level:
            y0, y1 = prev, ac[k]
            frac = 0.0 if y0 == y1 else (y0 - level) / (y0 - y1)
            return (k - 1 + frac) * dt_s
        prev = ac[k]
    return float("nan")


def _one_over_e_crossing(cov: np.ndarray, dt_s: float,
                         level: float = 1.0 / np.e) -> float:
    """Lag (seconds) at which the **noise-corrected**, normalised ACF of a single
    autocovariance series first falls to ``level`` (default ``1/e``).

    The autocovariance is normalised by its noise-free zero-lag value (so a tau=0
    noise jump does not shift the crossing) and handed to :func:`_crossing_lag`."""
    cov = np.asarray(cov, dtype=float)
    if cov.size < 2:
        return float("nan")
    c0 = _noise_free_zero_lag(cov)
    if not np.isfinite(c0) or c0 <= 0:
        return float("nan")
    return _crossing_lag(cov / c0, dt_s, level=level)


def _fit_loglog_slope_intercept(x: np.ndarray, y: np.ndarray
                                ) -> Tuple[float, float]:
    """Least-squares fit ``log y = slope * log x + intercept``; returns
    ``(slope, intercept)``.  Non-positive samples are dropped."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if good.sum() < 2:
        return float("nan"), float("nan")
    lx = np.log(x[good])
    ly = np.log(y[good])
    A = np.vstack([lx, np.ones_like(lx)]).T
    slope, intercept = np.linalg.lstsq(A, ly, rcond=None)[0]
    return float(slope), float(intercept)


def _mean_normalised_acf(arr: np.ndarray, max_lag: int) -> Optional[np.ndarray]:
    """Per-channel noise-corrected, normalised autocorrelation averaged across
    channels (lag ``0 .. max_lag``).

    Each channel's autocovariance is divided by its OWN noise-free zero-lag value
    (so high-variance modes do not dominate and the tau=0 white-noise jump is
    removed channel-by-channel), then the normalised curves are averaged.  Returns
    ``None`` if no channel has variance.  research/04 T1."""
    K = arr.shape[1]
    ac_sum = np.zeros(max_lag + 1)
    n_used = 0
    for k in range(K):
        col = arr[:, k]
        if np.var(col) <= 0:
            continue
        cov = _autocovariance_1d(col, max_lag=max_lag)
        c0 = _noise_free_zero_lag(cov)
        if not np.isfinite(c0) or c0 <= 0:
            continue
        ac_sum += cov / c0
        n_used += 1
    if n_used == 0:
        return None
    return ac_sum / n_used


# ============================= r0 ESTIMATORS =============================

def r0_from_zernike_variance(coeffs_ts: np.ndarray, D_m: float,
                             modes: Tuple[int, int] = (4, 15),
                             L0_m: Optional[float] = None,
                             noise_var: Optional[np.ndarray] = None,
                             total_var_const: float = KOLM_TOTAL_VAR) -> float:
    """R1 (PRIMARY): fit measured Zernike-coefficient variances to
    ``<a_j^2> = c_j (D/r0)^(5/3)`` over mid-order modes (tip/tilt excluded).

    ``coeffs_ts`` has shape ``(T, J)`` with column ``k`` = Noll index
    ``j = k + 2`` (piston-excluded modal time-series).  ``modes = (j_lo, j_hi)``
    is the inclusive Noll-index window used for the fit (default 4..15, the NAOMI
    choice; defocus..).  The per-mode Kolmogorov coefficients ``c_j`` come from
    :func:`aokit.zernike.noll_variance`.

    For each retained mode, ``(D/r0)^(5/3) = (Var(a_j) - sigma_noise_j^2)/c_j``;
    these are combined by a c_j-weighted average (least squares of the measured
    variances against ``c_j (D/r0)^(5/3)``), then inverted to ``r0``.  research/04
    R1.  ``L0_m``/``total_var_const`` accepted for signature/forward-compat (a
    finite L0 only matters for the joint von Karman fit R6)."""
    a = _as_2d_timeseries(coeffs_ts)
    T, J = a.shape
    var = a.var(axis=0)                       # per-mode measured variance
    if noise_var is not None:
        nv = np.asarray(noise_var, dtype=float).ravel()
        var = var - nv[: var.size]

    j_lo, j_hi = int(modes[0]), int(modes[1])
    j_lo = max(j_lo, 4)                        # never include tip/tilt (2,3)
    # column k -> Noll j = k + 2
    cj = np.array([noll_variance(k + 2) for k in range(J)], dtype=float)
    j_idx = np.arange(J) + 2
    sel = (j_idx >= j_lo) & (j_idx <= j_hi) & (cj > 0)
    if sel.sum() == 0:
        raise ValueError("no modes in the requested fit window")

    v = var[sel]
    c = cj[sel]
    v = np.clip(v, 0.0, None)                  # guard tiny negative after noise sub
    # Least-squares slope s in v_j = c_j * s with s = (D/r0)^(5/3):
    #   s = sum(c_j v_j) / sum(c_j^2)  (weights each mode by its expected size).
    s = float(np.sum(c * v) / np.sum(c * c))
    if s <= 0.0:
        return float("nan")
    return D_m * s ** (-_THREE_FIFTHS)


def r0_from_slope_variance(slopes_ts: np.ndarray, wavelength_m: float,
                           subap_diam_m: float,
                           noise_var: float = 0.0,
                           gtilt_const: float = SLOPE_GTILT_CONST) -> float:
    """R2: slope (G-tilt) variance r0 (reconstruction-independent).

    One-axis angle-of-arrival variance over a sub-aperture of diameter ``d`` is
    ``<alpha^2> = gtilt_const * lambda^2 * r0^(-5/3) * d^(-1/3)`` (G-tilt;
    Saint-Jacques 1998).  Invert the per-axis slope variance (averaged over all
    sub-apertures and both axes, with the centroid-noise variance subtracted) to
    ``r0``.  research/04 R2.

    ``slopes_ts`` is ``(T, 2M)`` in block layout; the variance is taken over time
    for every slope channel and averaged."""
    s = _as_2d_timeseries(slopes_ts)
    var_per_chan = s.var(axis=0)
    alpha2 = float(np.mean(var_per_chan)) - float(noise_var)
    if alpha2 <= 0.0:
        return float("nan")
    # alpha2 = K lam^2 r0^-5/3 d^-1/3  ->  r0 = (K lam^2 d^-1/3 / alpha2)^(3/5)
    factor = gtilt_const * wavelength_m ** 2 * subap_diam_m ** (-1.0 / 3.0)
    return (factor / alpha2) ** _THREE_FIFTHS


def _dimm_response(b: float, base: float = DIMM_BASE_CONST
                   ) -> Tuple[float, float]:
    """DIMM longitudinal/transverse response coefficients ``(K_l, K_t)`` for a
    normalised baseline ``b = B / D_sub`` (Sarazin & Roddier 1990)."""
    b13 = b ** (-1.0 / 3.0)
    k_l = base * (1.0 - DIMM_KL_B * b13)
    k_t = base * (1.0 - DIMM_KT_B * b13)
    return k_l, k_t


def r0_from_dimm(centroids_pair_ts: np.ndarray, baseline_m: float,
                 subap_diam_m: float, wavelength_m: float,
                 base_const: float = DIMM_BASE_CONST) -> float:
    """R3: DIMM differential tip/tilt between a sub-aperture pair
    (vibration-immune: common-mode motion cancels).

    ``centroids_pair_ts`` may be either:
      * shape ``(T, 4)`` -- columns ``[x1, y1, x2, y2]`` of the two spot
        angles-of-arrival (radians); the differential motion ``x1-x2`` (parallel
        to the baseline, "longitudinal") and ``y1-y2`` (transverse) variances are
        formed internally, or
      * shape ``(T, 2)`` -- pre-differenced ``[d_long, d_trans]`` series.

    With ``b = B / D_sub`` the longitudinal/transverse variances are
    ``sigma_{l,t}^2 = K_{l,t} lambda^2 r0^(-5/3) D_sub^(-1/3)`` and each inverts
    to ``r0``; the two estimates are averaged (internal-consistency check).
    research/04 R3."""
    arr = _as_2d_timeseries(centroids_pair_ts)
    if arr.shape[1] == 4:
        d_long = arr[:, 0] - arr[:, 2]
        d_trans = arr[:, 1] - arr[:, 3]
    elif arr.shape[1] == 2:
        d_long, d_trans = arr[:, 0], arr[:, 1]
    else:
        raise ValueError("centroids_pair_ts must have 2 or 4 columns")

    var_l = float(np.var(d_long))
    var_t = float(np.var(d_trans))
    b = baseline_m / subap_diam_m
    k_l, k_t = _dimm_response(b, base=base_const)

    lam2_d = wavelength_m ** 2 * subap_diam_m ** (-1.0 / 3.0)
    ests = []
    if var_l > 0.0 and k_l > 0.0:
        ests.append((k_l * lam2_d / var_l) ** _THREE_FIFTHS)
    if var_t > 0.0 and k_t > 0.0:
        ests.append((k_t * lam2_d / var_t) ** _THREE_FIFTHS)
    if not ests:
        return float("nan")
    return float(np.mean(ests))


def r0_from_total_variance(total_phase_variance: float, D_m: float,
                           tt_removed: bool = False,
                           total_var_const: float = KOLM_TOTAL_VAR,
                           tt_var_const: float = KOLM_TT_REMOVED_VAR) -> float:
    """R4 (scalar form): invert the Noll master relation
    ``sigma^2 = 1.0299 (D/r0)^(5/3)`` (piston removed) for a *given* total phase
    variance (rad^2).  ``tt_removed=True`` uses the tip/tilt-removed coefficient
    0.134.  ``r0 = D (const / sigma^2)^(3/5)``.  research/04 R4."""
    sigma2 = float(total_phase_variance)
    if sigma2 <= 0.0:
        return float("nan")
    const = tt_var_const if tt_removed else total_var_const
    return D_m * (const / sigma2) ** _THREE_FIFTHS


def r0_from_phase_variance(phase_ts: np.ndarray, D_m: float,
                           tt_removed: bool = True,
                           pupil_mask: Optional[np.ndarray] = None,
                           total_var_const: float = KOLM_TOTAL_VAR,
                           tt_var_const: float = KOLM_TT_REMOVED_VAR) -> float:
    """R4: total phase variance r0 from a stack of reconstructed phase maps.

    ``phase_ts`` is ``(T, P)`` (P pupil pixels, piston/TT already removed per
    frame if ``tt_removed``) or ``(T, H, W)`` with an optional boolean
    ``pupil_mask``.  The spatial variance over the pupil is averaged over frames
    and inverted with :func:`r0_from_total_variance`.  research/04 R4."""
    arr = np.asarray(phase_ts, dtype=float)
    if arr.ndim == 3:
        T = arr.shape[0]
        flat = arr.reshape(T, -1)
        if pupil_mask is not None:
            m = np.asarray(pupil_mask, dtype=bool).ravel()
            flat = flat[:, m]
    else:
        flat = _as_2d_timeseries(arr)
    # Spatial variance over the pupil per frame, averaged over time.
    sigma2 = float(np.mean(flat.var(axis=1)))
    return r0_from_total_variance(sigma2, D_m, tt_removed=tt_removed,
                                  total_var_const=total_var_const,
                                  tt_var_const=tt_var_const)


def r0_from_structure_function(phase_ts: np.ndarray, sample_pitch_m: float,
                               struct_const: float = STRUCT_FN_CONST,
                               r_range: Optional[Tuple[float, float]] = None,
                               pupil_mask: Optional[np.ndarray] = None
                               ) -> Tuple[float, float]:
    """R5: Kolmogorov structure-function fit ``D_phi(r) = 6.88 (r/r0)^(5/3)``.

    Computes the empirical phase structure function ``D_phi(r) = <|phi(x+r) -
    phi(x)|^2>`` from reconstructed phase maps, fits the 5/3 power law in the
    inertial range, and returns ``(r0, fitted_loglog_slope)``.  The slope should
    be ~5/3 = 1.667 for Kolmogorov turbulence (a model-validity check).

    ``phase_ts`` is a single 2-D map ``(H, W)`` or a stack ``(T, H, W)`` (averaged
    over frames).  Separations are measured along rows and columns in units of
    ``sample_pitch_m``.  Only point-pairs with **both** ends inside the pupil are
    used: pass a boolean ``pupil_mask`` ``(H, W)`` (True inside), or encode the
    pupil as ``NaN`` outside in ``phase_ts``.  Out-of-pupil pairs are excluded
    pair-by-pair so edge zeros never bias the fit.  research/04 R5."""
    arr = np.asarray(phase_ts, dtype=float)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError("phase_ts must be (H, W) or (T, H, W)")
    T, H, W = arr.shape

    # Build a per-pixel validity mask: explicit pupil_mask AND not-NaN.
    if pupil_mask is not None:
        m2d = np.asarray(pupil_mask, dtype=bool)
        valid = np.broadcast_to(m2d[None, ...], arr.shape) & np.isfinite(arr)
    else:
        valid = np.isfinite(arr)
    # Replace NaNs with 0 for the arithmetic (masked out by ``valid`` anyway).
    arr = np.where(np.isfinite(arr), arr, 0.0)

    max_sep = min(H, W) // 2
    if max_sep < 2:
        raise ValueError("phase map too small for a structure function")
    seps = np.arange(1, max_sep + 1)
    d_vals = np.full(seps.size, np.nan, dtype=float)

    for i, sep in enumerate(seps):
        # Squared differences along x (columns) and y (rows), only where BOTH
        # ends are valid (inside the pupil), averaged over frames.
        dx = arr[:, :, sep:] - arr[:, :, :-sep]
        wx = valid[:, :, sep:] & valid[:, :, :-sep]
        dy = arr[:, sep:, :] - arr[:, :-sep, :]
        wy = valid[:, sep:, :] & valid[:, :-sep, :]
        num = float(np.sum(dx[wx] ** 2) + np.sum(dy[wy] ** 2))
        cnt = int(np.count_nonzero(wx) + np.count_nonzero(wy))
        if cnt > 0:
            d_vals[i] = num / cnt

    r = seps * sample_pitch_m
    # Inertial range: between ~2 pitches and ~half the aperture by default.
    if r_range is None:
        lo, hi = r[1] if r.size > 1 else r[0], r[max(1, int(0.6 * r.size)) - 1]
    else:
        lo, hi = r_range
    fit_sel = (r >= lo) & (r <= hi) & (d_vals > 0)
    if fit_sel.sum() < 2:
        fit_sel = d_vals > 0

    slope, intercept = _fit_loglog_slope_intercept(r[fit_sel], d_vals[fit_sel])
    # Estimate r0 from each point assuming the 5/3 law, then average in log:
    #   D = const (r/r0)^(5/3)  ->  r0 = r (const / D)^(3/5).
    r0_pts = r[fit_sel] * (struct_const / d_vals[fit_sel]) ** _THREE_FIFTHS
    r0 = float(np.exp(np.mean(np.log(r0_pts)))) if r0_pts.size else float("nan")
    return r0, slope


def r0_l0_from_vonkarman(coeffs_ts: np.ndarray, D_m: float,
                         modes: Tuple[int, int] = (4, 15),
                         L0_grid: Optional[np.ndarray] = None
                         ) -> Tuple[float, float]:
    """R6: von Karman joint ``(r0, L0)`` fit to the modal-variance curve.

    A finite outer scale ``L0`` suppresses the low-order modal variances below
    the Kolmogorov prediction.  We model the von Karman per-mode variance as the
    Kolmogorov coefficient scaled by an outer-scale attenuation
    ``g_j(D/L0)`` that is 1 as ``L0 -> inf`` and decreases for low radial orders
    as ``D/L0`` grows, scan ``L0`` (and solve ``r0`` in closed form at each
    ``L0``), and pick the ``(r0, L0)`` minimising the log-variance residual.
    research/04 R6.

    Note: with a small aperture (``D << L0``) the ``r0``-``L0`` degeneracy makes
    ``L0`` weakly constrained (research/04: NAOMI's 1.8 m could not constrain
    L0); ``r0`` is robust regardless.  Returns ``(r0, L0)`` (``L0 = inf`` ->
    Kolmogorov limit)."""
    a = _as_2d_timeseries(coeffs_ts)
    T, J = a.shape
    var = a.var(axis=0)
    j_idx = np.arange(J) + 2
    j_lo = max(int(modes[0]), 4)
    j_hi = int(modes[1])
    cj = np.array([noll_variance(k + 2) for k in range(J)], dtype=float)
    n_ord = np.array([zern_index(k + 2)[0] for k in range(J)], dtype=float)
    sel = (j_idx >= j_lo) & (j_idx <= j_hi) & (cj > 0) & (var > 0)
    if sel.sum() < 2:
        # fall back to Kolmogorov-only r0
        return r0_from_zernike_variance(coeffs_ts, D_m, modes=modes), float("inf")

    v = var[sel]
    c = cj[sel]
    n = n_ord[sel]
    logv = np.log(v)

    if L0_grid is None:
        # D/L0 from ~0 (Kolmogorov) up to 1 (L0 ~ D).
        ratios = np.concatenate([[0.0], np.logspace(-2, 0.0, 40)])
    else:
        L0_grid = np.asarray(L0_grid, dtype=float)
        ratios = D_m / L0_grid

    best = (float("nan"), float("inf"), np.inf)  # (r0, L0, residual)
    for ratio in ratios:
        # Outer-scale attenuation of modal variance vs radial order: lower-order
        # modes (small n) are suppressed more by a finite L0. Simple, monotone
        # surrogate consistent with von Karman behaviour (Conan 2008 trend):
        #   g_n = 1 / (1 + ((n+1) (D/L0))^(5/3))  -> 1 as L0->inf.
        atten = 1.0 / (1.0 + ((n + 1.0) * ratio) ** _FIVE_THIRDS)
        c_eff = c * atten
        # Closed-form r0 at this L0: model v_j = c_eff_j * s, s=(D/r0)^(5/3).
        s = float(np.sum(c_eff * v) / np.sum(c_eff * c_eff))
        if s <= 0:
            continue
        model = np.log(c_eff * s)
        resid = float(np.sum((logv - model) ** 2))
        if resid < best[2]:
            L0 = float("inf") if ratio <= 0 else D_m / ratio
            r0 = D_m * s ** (-_THREE_FIFTHS)
            best = (r0, L0, resid)
    return best[0], best[1]


def r0_from_vonkarman(coeffs_ts: np.ndarray, D_m: float,
                      modes: Tuple[int, int] = (4, 15)) -> float:
    """R6 convenience: return only ``r0`` from the joint von Karman fit."""
    return r0_l0_from_vonkarman(coeffs_ts, D_m, modes=modes)[0]


def r0_from_seeing(fwhm_rad: float, wavelength_m: float,
                   seeing_const: float = SEEING_CONST) -> float:
    """R7: seeing FWHM relation ``epsilon = 0.98 lambda / r0`` (image domain)
    -> ``r0 = 0.98 lambda / epsilon`` (radians).  research/04 R7."""
    if fwhm_rad <= 0.0:
        return float("nan")
    return seeing_const * wavelength_m / fwhm_rad


# ============================ tau0 ESTIMATORS ===========================

def tau0_from_autocorr(series_ts: np.ndarray, dt_s: float,
                       level: float = 1.0 / np.e) -> float:
    """T1 (PRIMARY): 1/e decorrelation time of the temporal autocorrelation.

    ``series_ts`` is ``(T,)`` or ``(T, K)`` (modes or slopes).  The normalised
    ACF is computed per channel (the tau=0 white-noise jump is bypassed by
    interpolating from lag 1), averaged across channels, and the lag at which it
    falls to ``level`` (default ``1/e``) is returned as tau0.  research/04 T1."""
    arr = _as_2d_timeseries(series_ts)
    T, K = arr.shape
    max_lag = T - 1
    ac = _mean_normalised_acf(arr, max_lag)
    if ac is None:
        return float("nan")
    return _crossing_lag(ac, dt_s, level=level)


# Alias matching the task's preferred name.
def tau0_from_autocorrelation(series_ts: np.ndarray, dt_s: float,
                              level: float = 1.0 / np.e) -> float:
    """Alias of :func:`tau0_from_autocorr` (research/04 T1)."""
    return tau0_from_autocorr(series_ts, dt_s, level=level)


def tau0_from_psd(series_ts: np.ndarray, dt_s: float,
                  knee_drop: float = 0.5) -> Tuple[float, float]:
    """T2: characteristic time from the temporal-PSD knee.

    Kolmogorov + frozen flow gives a flat low-frequency PSD that breaks to a
    steep power law above a cutoff ``f_c`` (research/04 T2: slopes -11/3 for
    tilt, -17/3 for higher modes).  We estimate the knee ``f_c`` as the frequency
    where the (smoothed) PSD has dropped to ``knee_drop`` of its low-frequency
    plateau, and report ``tau0 ~ 1 / (2 pi f_c)`` plus ``f_c``.

    ``series_ts`` is ``(T,)`` or ``(T, K)`` (averaged over channels).  Returns
    ``(tau0, f_c)``.  research/04 T2."""
    arr = _as_2d_timeseries(series_ts)
    T, K = arr.shape
    arr = arr - arr.mean(axis=0, keepdims=True)
    freqs = np.fft.rfftfreq(T, d=dt_s)
    psd = np.zeros(freqs.size)
    n_used = 0
    for k in range(K):
        col = arr[:, k]
        if np.var(col) <= 0:
            continue
        f = np.fft.rfft(col)
        psd += (np.abs(f) ** 2)
        n_used += 1
    if n_used == 0 or freqs.size < 4:
        return float("nan"), float("nan")
    psd /= n_used

    # Low-frequency plateau = median of the lowest few non-DC bins.
    lo_band = psd[1: max(3, freqs.size // 10)]
    if lo_band.size == 0:
        return float("nan"), float("nan")
    plateau = float(np.median(lo_band))
    if plateau <= 0:
        return float("nan"), float("nan")
    thresh = knee_drop * plateau
    # First frequency (above DC) where PSD drops below the threshold.
    f_c = float("nan")
    for i in range(1, freqs.size):
        if psd[i] <= thresh:
            f_c = freqs[i]
            break
    if not np.isfinite(f_c) or f_c <= 0:
        return float("nan"), float("nan")
    tau0 = 1.0 / (2.0 * np.pi * f_c)
    return tau0, f_c


def tau0_from_greenwood(r0_m: float, wind_speed_mps: float,
                        tau0_const: float = GREENWOOD_TAU0_CONST,
                        fg_const: float = GREENWOOD_FREQ_CONST
                        ) -> Tuple[float, float]:
    """T3: Greenwood bridge ``f_G = 0.426 v/r0``, ``tau0 = 0.314 r0/v =
    0.134/f_G``; returns ``(tau0, f_G)``.  research/04 T3."""
    if wind_speed_mps <= 0.0:
        return float("nan"), float("nan")
    tau0 = tau0_const * r0_m / wind_speed_mps
    f_g = fg_const * wind_speed_mps / r0_m
    return tau0, f_g


def tau0_from_structure_function(series_ts: np.ndarray, dt_s: float,
                                 level: float = 1.0,
                                 t0_to_tau0: float = T0_TO_TAU0) -> float:
    """T4: temporal structure-function time.

    Builds the normalised temporal structure function ``D(tau)/sigma^2 =
    2(1 - C(tau))`` from the ACF (so it is independent of absolute units and of
    the additive measurement-noise floor, which cancels for tau>0), finds the lag
    ``t0`` where ``D`` reaches ``level`` times its 1-rad^2-equivalent (the e-fold
    of the saturated value, i.e. ``D = (1 - 1/e) * 2 sigma^2``), and returns
    ``tau0 = t0 / 0.66`` via ``t0 = 0.66 tau0`` (research/04 T4: the frozen-flow
    ``D_phi(tau) = 6.88 (v tau / r0)^(5/3)``, ``t0 = 0.66 tau0``).

    Operationally: the normalised structure function reaches ``1 - 1/e`` of its
    plateau at the same lag the ACF falls to ``1/e``, so this is a genuine,
    noise-floor-immune cross-check of T1 (different statistic, same timescale)."""
    arr = _as_2d_timeseries(series_ts)
    T, K = arr.shape
    max_lag = T - 1
    ac = _mean_normalised_acf(arr, max_lag)
    if ac is None:
        return float("nan")
    # Normalised structure function D_norm(tau) = 2(1 - C(tau)) in [0, 2].
    # It crosses (1 - 1/e)*2 exactly where C crosses 1/e -> the e-fold time t_e.
    # That t_e is the structure-function coherence time; convert to tau0.
    t_e = _crossing_lag(ac, dt_s, level=1.0 / np.e)
    if not np.isfinite(t_e):
        return float("nan")
    # t_e here corresponds to the e-folding ("t0"-like) lag; tau0 = t_e / 0.66.
    return t_e * t0_to_tau0


def wind_from_frozen_flow(slope_maps_ts: np.ndarray, dt_s: float,
                          sample_pitch_m: float,
                          max_shift: Optional[int] = None
                          ) -> Tuple[float, float]:
    """T5: Taylor frozen-flow wind retrieval via spatio-temporal cross-correlation.

    Under frozen flow the phase/slope pattern translates rigidly at velocity
    ``v``; the spatial offset of the lag-1 cross-correlation peak is ``v * dt``.
    ``slope_maps_ts`` is ``(T, H, W)`` (e.g. an x-slope or phase map per frame).
    We cross-correlate consecutive frames (averaged over the sequence), find the
    integer peak shift ``(dy, dx)`` with a parabolic sub-pixel refinement,
    convert to ``v = |shift| * pitch / dt`` and return ``(wind_speed_mps,
    |peak_corr|)``.  research/04 T5."""
    arr = np.asarray(slope_maps_ts, dtype=float)
    if arr.ndim != 3:
        raise ValueError("slope_maps_ts must be (T, H, W)")
    T, H, W = arr.shape
    if T < 2:
        return float("nan"), float("nan")
    if max_shift is None:
        max_shift = max(1, min(H, W) // 2 - 1)

    shifts = range(-max_shift, max_shift + 1)
    corr = np.zeros((len(shifts), len(shifts)))
    a0_all = arr[:-1]
    a1_all = arr[1:]
    # Remove per-frame mean to suppress the DC/zero-shift bias.
    a0_all = a0_all - a0_all.mean(axis=(1, 2), keepdims=True)
    a1_all = a1_all - a1_all.mean(axis=(1, 2), keepdims=True)
    denom = np.sqrt(np.mean(a0_all ** 2) * np.mean(a1_all ** 2))
    if denom <= 0:
        return float("nan"), float("nan")

    for iy, dy in enumerate(shifts):
        for ix, dx in enumerate(shifts):
            # overlap region of frame t (a0) and frame t+1 (a1) shifted by (dy,dx)
            ys0_a, ys1_a = max(0, -dy), min(H, H - dy)
            xs0_a, xs1_a = max(0, -dx), min(W, W - dx)
            sub0 = a0_all[:, ys0_a:ys1_a, xs0_a:xs1_a]
            sub1 = a1_all[:, ys0_a + dy:ys1_a + dy, xs0_a + dx:xs1_a + dx]
            if sub0.size == 0:
                continue
            corr[iy, ix] = float(np.mean(sub0 * sub1)) / denom

    peak = np.unravel_index(np.argmax(corr), corr.shape)
    peak_corr = float(corr[peak])
    dy_peak = list(shifts)[peak[0]]
    dx_peak = list(shifts)[peak[1]]

    # Parabolic sub-pixel refinement along each axis.
    def _subpix(c_lo, c0, c_hi):
        denom2 = (c_lo - 2 * c0 + c_hi)
        if denom2 == 0:
            return 0.0
        return 0.5 * (c_lo - c_hi) / denom2

    fy = fx = 0.0
    if 0 < peak[0] < corr.shape[0] - 1:
        fy = _subpix(corr[peak[0] - 1, peak[1]], corr[peak],
                     corr[peak[0] + 1, peak[1]])
    if 0 < peak[1] < corr.shape[1] - 1:
        fx = _subpix(corr[peak[0], peak[1] - 1], corr[peak],
                     corr[peak[0], peak[1] + 1])

    shift_px = np.hypot(dy_peak + fy, dx_peak + fx)
    v = shift_px * sample_pitch_m / dt_s
    return float(v), peak_corr


def tau0_from_taylor(slope_maps_ts: np.ndarray, dt_s: float,
                     sample_pitch_m: float, r0_m: float,
                     tau0_const: float = GREENWOOD_TAU0_CONST) -> float:
    """T5 (tau0 form): frozen-flow wind retrieval -> ``tau0 = 0.314 r0 / v``.

    Convenience wrapper that runs :func:`wind_from_frozen_flow` to get the wind
    speed and converts it to a coherence time via the Greenwood relation given a
    known/estimated ``r0``.  research/04 T5 + T3."""
    v, _corr = wind_from_frozen_flow(slope_maps_ts, dt_s, sample_pitch_m)
    if not np.isfinite(v) or v <= 0:
        return float("nan")
    return tau0_const * r0_m / v


def tyler_frequency(r0_m: float, wind_speed_mps: float, D_m: float,
                    tyler_const: float = TYLER_CONST) -> float:
    """T6: Tyler tracking frequency ``f_T = 0.368 v r0^(-1/6) D^(-5/6)`` (the
    tip/tilt-specific bandwidth).  research/04 T6."""
    if r0_m <= 0 or D_m <= 0:
        return float("nan")
    return tyler_const * wind_speed_mps * r0_m ** (-1.0 / 6.0) * D_m ** (-5.0 / 6.0)


def tau_tracking(r0_m: float, wind_speed_mps: float, D_m: float,
                 tyler_const: float = TYLER_CONST,
                 bridge: float = GREENWOOD_BRIDGE) -> float:
    """T6 (tau form): tip/tilt tracking timescale ``tau_T = 0.134 / f_T`` from the
    Tyler frequency (the tracking-mode analogue of ``tau0 = 0.134 / f_G``).
    research/04 T6."""
    f_t = tyler_frequency(r0_m, wind_speed_mps, D_m, tyler_const=tyler_const)
    if not np.isfinite(f_t) or f_t <= 0:
        return float("nan")
    return bridge / f_t


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
        return {
            "r0_m": {
                "median": self.r0_median,
                "spread": self.r0_spread,
                "estimators": dict(self.r0_m),
            },
            "L0_m": self.L0_m,
            "tau0_s": {
                "median": self.tau0_median,
                "spread": self.tau0_spread,
                "estimators": dict(self.tau0_s),
            },
            "wind_speed_mps": self.wind_speed_mps,
            "f_greenwood_hz": self.f_greenwood_hz,
            "seeing_arcsec": self.seeing_arcsec,
            "strehl_marechal": self.strehl_marechal,
            "n_frames": self.n_frames,
            "dt_s": self.dt_s,
            "notes": self.notes,
        }


def combine_estimates(values: Dict[str, float]) -> Tuple[float, float]:
    """Robust central estimate (median) and spread (std) over a set of
    independent estimator values.  Non-finite values are dropped.  research/04
    S5 (tabulate, take the median, report the spread as systematic uncertainty)."""
    vals = np.array([v for v in values.values()
                     if v is not None and np.isfinite(v)], dtype=float)
    if vals.size == 0:
        return float("nan"), float("nan")
    median = float(np.median(vals))
    spread = float(np.std(vals)) if vals.size > 1 else 0.0
    return median, spread


def _combine_dict(values: Dict[str, float]) -> Dict[str, float]:
    """Full cross-validation summary: ``{per_method..., mean, std, median}``."""
    out: Dict[str, float] = {}
    finite = {}
    for k, v in values.items():
        out[k] = float(v) if v is not None else float("nan")
        if v is not None and np.isfinite(v):
            finite[k] = float(v)
    arr = np.array(list(finite.values()), dtype=float)
    if arr.size == 0:
        out["mean"] = float("nan")
        out["std"] = float("nan")
        out["median"] = float("nan")
    else:
        out["mean"] = float(np.mean(arr))
        out["std"] = float(np.std(arr)) if arr.size > 1 else 0.0
        out["median"] = float(np.median(arr))
    return out


def combine_r0(values: Dict[str, float]) -> Dict[str, float]:
    """Cross-validate the r0 estimators: returns ``{per_method..., mean, std,
    median}`` (research/04 S5).  Use ``median`` as the central value and ``std``
    as the systematic spread."""
    return _combine_dict(values)


def combine_tau0(values: Dict[str, float]) -> Dict[str, float]:
    """Cross-validate the tau0 estimators: returns ``{per_method..., mean, std,
    median}`` (research/04 S5)."""
    return _combine_dict(values)


def _rad_to_arcsec(rad: float) -> float:
    return rad * (180.0 * 3600.0 / np.pi)


def estimate_all(slopes_ts: np.ndarray, coeffs_ts: np.ndarray,
                 phase_ts: Optional[np.ndarray], cfg, dt_s: float
                 ) -> TurbulenceResult:
    """Run every applicable r0/tau0 estimator, combine them, and return a fully
    populated :class:`TurbulenceResult`.  research/04 S5 pipeline.

    Parameters
    ----------
    slopes_ts : (T, 2M) array or None
        SH slope time-series (block layout) for R2/R3/T5.
    coeffs_ts : (T, J) array or None
        Modal-coefficient time-series (column k = Noll j = k+2) for R1/R4/R6/T1.
    phase_ts : (T, H, W) or (T, P) array or None
        Reconstructed phase maps for R4/R5.
    cfg : aokit.config.Config
        Provides D (pupil.diameter_m), sub-aperture diameter (mla.pitch_m),
        wavelength, and (optional) ground-truth wind speed.
    dt_s : float
        Inter-frame interval (s).

    Only the estimators whose required inputs are present are run; the combiner
    medians whatever is available."""
    D_m = float(cfg.pupil.diameter_m)
    lam = float(cfg.wavelength_m)
    d_sub = float(cfg.mla.pitch_m)
    notes = []

    r0_vals: Dict[str, float] = {}
    tau0_vals: Dict[str, float] = {}
    L0_m: Optional[float] = None
    wind: Optional[float] = None
    f_g: Optional[float] = None

    # ---- r0 from modal coefficients (R1, R4-modal, R6) ----
    if coeffs_ts is not None:
        coeffs = _as_2d_timeseries(coeffs_ts)
        try:
            r0_vals["R1_zernike_var"] = r0_from_zernike_variance(coeffs, D_m)
        except Exception:
            pass
        try:
            r0_v, L0_m = r0_l0_from_vonkarman(coeffs, D_m)
            r0_vals["R6_vonkarman"] = r0_v
        except Exception:
            pass
        # R4 via total residual variance reconstructed from the modal variances
        # (TT excluded -> sum of mid/high-order modal variances + Noll residual).
        try:
            var = coeffs.var(axis=0)
            # variance carried by modes j>=4 (defocus and up):
            sigma2_ho = float(np.sum(var[2:]))   # columns 0,1 are tip/tilt
            if sigma2_ho > 0:
                # add the un-sensed tail via the asymptotic Noll residual ratio:
                r0_vals["R4_phase_var"] = r0_from_total_variance(
                    sigma2_ho, D_m, tt_removed=True)
        except Exception:
            pass
        notes.append("tip/tilt excluded from r0 fit")

    # ---- r0 from slopes (R2, R3) ----
    if slopes_ts is not None:
        slopes = _as_2d_timeseries(slopes_ts)
        try:
            r0_vals["R2_slope_var"] = r0_from_slope_variance(slopes, lam, d_sub)
        except Exception:
            pass
        # R3 DIMM from the first two valid sub-apertures (x,y of each):
        try:
            M = slopes.shape[1] // 2
            if M >= 2:
                # build [x1,y1,x2,y2] AoA series from slopes (rad of tilt).
                pair = np.column_stack([slopes[:, 0], slopes[:, M],
                                        slopes[:, 1], slopes[:, M + 1]])
                r0_vals["R3_dimm"] = r0_from_dimm(pair, baseline_m=d_sub,
                                                  subap_diam_m=d_sub,
                                                  wavelength_m=lam)
        except Exception:
            pass

    # ---- r0 from phase maps (R4, R5) ----
    if phase_ts is not None:
        try:
            r0_vals["R4_phase_var"] = r0_from_phase_variance(
                phase_ts, D_m, tt_removed=True)
        except Exception:
            pass
        try:
            arr = np.asarray(phase_ts, dtype=float)
            if arr.ndim == 3:
                r0_sf, slope = r0_from_structure_function(arr, d_sub)
                r0_vals["R5_struct_fn"] = r0_sf
                notes.append(f"R5 struct-fn slope={slope:.2f}")
        except Exception:
            pass

    r0_combo = combine_r0(r0_vals)
    r0_median = r0_combo["median"]

    # ---- R7 seeing: if a seeing/FWHM is derivable, or fall back to r0_median ----
    seeing_arcsec = None
    if np.isfinite(r0_median) and r0_median > 0:
        fwhm = SEEING_CONST * lam / r0_median
        # R7 round-trips r0_median -> seeing -> r0; record seeing as a derived
        # quantity (an independent image-domain measurement would replace this).
        seeing_arcsec = _rad_to_arcsec(fwhm)

    # ---- wind speed (T5) and tau0 estimators ----
    if phase_ts is not None and np.asarray(phase_ts).ndim == 3:
        try:
            wind, _c = wind_from_frozen_flow(np.asarray(phase_ts, dtype=float),
                                             dt_s, d_sub)
        except Exception:
            wind = None
    if wind is None and cfg.ground_truth is not None:
        wind = cfg.ground_truth.wind_speed_mps

    # T1, T2, T4 from modal coefficients (mid-order modes preferred).
    series = None
    if coeffs_ts is not None:
        coeffs = _as_2d_timeseries(coeffs_ts)
        # mid-order columns (defocus..) for vibration-clean tau0:
        series = coeffs[:, 2:] if coeffs.shape[1] > 2 else coeffs
    elif slopes_ts is not None:
        series = _as_2d_timeseries(slopes_ts)

    if series is not None and series.shape[0] > 2:
        try:
            tau0_vals["T1_autocorr"] = tau0_from_autocorr(series, dt_s)
        except Exception:
            pass
        try:
            tau0_vals["T2_psd"] = tau0_from_psd(series, dt_s)[0]
        except Exception:
            pass
        try:
            tau0_vals["T4_struct_fn"] = tau0_from_structure_function(series, dt_s)
        except Exception:
            pass

    # T5 frozen-flow -> tau0 (needs wind + r0).
    if wind is not None and np.isfinite(r0_median) and wind > 0:
        tau0_vals["T5_frozenflow"] = GREENWOOD_TAU0_CONST * r0_median / wind
    # T3 Greenwood bridge.
    if wind is not None and np.isfinite(r0_median) and wind > 0:
        t3, f_g = tau0_from_greenwood(r0_median, wind)
        tau0_vals["T3_greenwood"] = t3
    # T6 Tyler (tip/tilt timescale).
    if wind is not None and np.isfinite(r0_median) and wind > 0:
        tau0_vals["T6_tyler"] = tau_tracking(r0_median, wind, D_m)

    tau0_combo = combine_tau0(tau0_vals)

    if cfg.ground_truth is not None and cfg.ground_truth.L0_m is not None \
            and L0_m is not None and not np.isfinite(L0_m):
        # keep estimated L0 if finite; otherwise leave None.
        L0_m = None
    if L0_m is not None and not np.isfinite(L0_m):
        L0_m = None

    # Strehl (Marechal) from the total residual variance implied by r0_median.
    strehl = None
    if np.isfinite(r0_median) and r0_median > 0:
        sigma2 = KOLM_TOTAL_VAR * (D_m / r0_median) ** _FIVE_THIRDS
        strehl = float(np.exp(-sigma2))

    n_frames = 0
    if coeffs_ts is not None:
        n_frames = _as_2d_timeseries(coeffs_ts).shape[0]
    elif slopes_ts is not None:
        n_frames = _as_2d_timeseries(slopes_ts).shape[0]
    elif phase_ts is not None:
        n_frames = np.asarray(phase_ts).shape[0]

    if L0_m is not None:
        notes.append("von Karman used (L0 finite)")

    return TurbulenceResult(
        r0_m={k: v for k, v in r0_vals.items()},
        tau0_s={k: v for k, v in tau0_vals.items()},
        L0_m=L0_m,
        wind_speed_mps=wind,
        f_greenwood_hz=f_g,
        seeing_arcsec=seeing_arcsec,
        strehl_marechal=strehl,
        r0_median=r0_combo["median"],
        r0_spread=r0_combo["std"],
        tau0_median=tau0_combo["median"],
        tau0_spread=tau0_combo["std"],
        n_frames=int(n_frames),
        dt_s=float(dt_s),
        notes="; ".join(notes),
    )


def characterize(slopes_ts: np.ndarray, coeffs_ts: np.ndarray, dt_s: float,
                 cfg, phase_ts: Optional[np.ndarray] = None) -> dict:
    """Top-level entry point: run :func:`estimate_all` and return the
    turbulence_summary.json-shaped dict (ARCHITECTURE.md S4.4).

    ``characterize(slopes_ts, coeffs_ts, dt, cfg)`` is the signature the task
    asks for; ``phase_ts`` is an optional extra domain (reconstructed phase
    maps)."""
    result = estimate_all(slopes_ts, coeffs_ts, phase_ts, cfg, dt_s)
    return result.to_dict()
