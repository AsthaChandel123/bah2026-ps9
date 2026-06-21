"""aokit.validation -- correctness & quality metrics (recovered vs. true).

RMS WFE, Strehl (Marechal), residual variance, phase correlation, r0/tau0
recovery error, reconstructor self-consistency, DM-corrected residual,
Zernike-coefficient error, an error-budget summary, and C/Python parity.
research/07 PART C; ARCHITECTURE.md S8.2.

Sign / unit conventions (fixed and documented here)
---------------------------------------------------
* **Phase units** -- ``phase`` maps and residual standard deviations are in
  **radians** unless a name says otherwise.  The wavelength helper
  :func:`rad_to_meters` converts an RMS in radians to metres of optical path
  via ``OPD = (lambda / 2pi) * sigma_rad``.

* **Piston removal** -- every variance/RMS/Strehl metric removes the mean
  (piston) over the valid pupil first, because piston is unobservable and does
  not affect image quality.  Tilt removal is optional (off by default for the
  single-map helpers, on for the two-map ``rms_wfe`` to match the
  "piston/tip-tilt removed" pass criterion in ARCHITECTURE.md S8.2).

* **Strehl (extended Marechal)** -- ``S = exp(-sigma_phi_res^2)`` with
  ``sigma_phi_res`` the *residual* RMS phase in radians (research/07 C.1).
  Accepts either a residual phase map (its piston-removed RMS is used) or a
  pre-computed residual **variance** scalar.

* **DM reflection (the -1/2 convention)** -- the DM *command* matrix is built
  (``aokit.dm.build_command_matrix``) so that the mechanical mirror surface
  ``H @ commands`` approximates the **conjugate** target ``-phi/2``: a reflective
  DM imprints **twice** its surface displacement onto the wavefront, with a sign
  flip, so the wavefront correction it applies is ``W_dm = -2 * (H @ commands)``.
  The corrected residual is therefore

      W_res = W_true - W_dm = W_true + 2 * (H @ commands).

  Hence if ``commands == -W_true/2`` (and ``H`` is the identity, i.e. perfect
  fitting) the residual is exactly zero.  :func:`residual_after_dm` accepts the
  commands and ``H`` and applies this convention; :func:`dm_residual` accepts an
  already-realised ``dm_surface`` (the wavefront correction ``W_dm`` itself) and
  simply differences it.
"""
from __future__ import annotations

from typing import Optional, Tuple, Union
import numpy as np


# ============================================================================
# Internal helpers
# ============================================================================

def _as_masked_flat(arr: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """Flatten ``arr`` to the valid pupil pixels as float64.

    ``mask`` (if given) is broadcast/ravelled to bool and selects the pixels;
    otherwise all finite pixels are kept (NaNs outside a pupil are dropped).
    """
    a = np.asarray(arr, dtype=np.float64).ravel()
    if mask is not None:
        m = np.asarray(mask, dtype=bool).ravel()
        if m.shape != a.shape:
            raise ValueError(
                f"mask shape {m.shape} incompatible with data shape {a.shape}"
            )
        a = a[m]
    # Drop non-finite samples (e.g. NaN-padded outside-pupil regions).
    finite = np.isfinite(a)
    if not finite.all():
        a = a[finite]
    return a


def _remove_piston(a: np.ndarray) -> np.ndarray:
    """Subtract the mean (piston) over the supplied samples."""
    if a.size == 0:
        return a
    return a - a.mean()


# ============================================================================
# 1. RMS wavefront error
# ============================================================================

def rms_wfe(W_recon: np.ndarray, W_true: Optional[np.ndarray] = None,
            mask: Optional[np.ndarray] = None,
            remove_piston: bool = True, remove_tilt: bool = False) -> float:
    """RMS wavefront error over the valid pupil, in **radians** (research/07 C.1).

    Two call forms:

    * ``rms_wfe(phase, mask=...)`` -- RMS of a single phase map (the WFE of the
      map itself, e.g. an uncorrected or residual wavefront).
    * ``rms_wfe(W_recon, W_true, mask=...)`` -- RMS of the *error*
      ``W_recon - W_true`` (the reconstruction error,
      ``sigma_WFE = sqrt(mean((W_recon - W_true)^2))``).

    Piston is removed by default (it is unobservable).  ``remove_tilt`` also
    drops the best-fit plane (mean x/y gradient) before measuring; it requires a
    2-D map and is off by default for the single-map form.

    Convert to metres of optical path with :func:`rad_to_meters`.
    """
    if W_true is None:
        err = np.asarray(W_recon, dtype=np.float64)
    else:
        err = np.asarray(W_recon, dtype=np.float64) - np.asarray(
            W_true, dtype=np.float64)

    if remove_tilt:
        err = _remove_plane(err, mask)

    a = _as_masked_flat(err, mask)
    if a.size == 0:
        return 0.0
    if remove_piston:
        a = _remove_piston(a)
    return float(np.sqrt(np.mean(a * a)))


def _remove_plane(W: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """Subtract the least-squares best-fit plane (piston + tip + tilt) from a
    2-D phase map over the valid pupil.  Returns a map of the same shape with
    the fitted plane removed inside the mask (outside is left unchanged)."""
    W = np.asarray(W, dtype=np.float64)
    if W.ndim != 2:
        # Cannot fit an x/y plane to a non-2-D array; just remove piston.
        flat = W.ravel().astype(np.float64)
        return (W - flat[np.isfinite(flat)].mean()) if flat.size else W
    ny, nx = W.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
    else:
        m = np.isfinite(W)
    xs = xx[m].astype(np.float64)
    ys = yy[m].astype(np.float64)
    zs = W[m].astype(np.float64)
    if zs.size < 3:
        out = W.copy()
        if zs.size:
            out[m] = zs - zs.mean()
        return out
    A = np.column_stack([np.ones_like(xs), xs, ys])
    coef, *_ = np.linalg.lstsq(A, zs, rcond=None)
    out = W.copy()
    out[m] = zs - A @ coef
    return out


def rad_to_meters(sigma_rad: float, wavelength_m: float) -> float:
    """Convert an RMS phase in radians to RMS optical-path-difference in metres.

    ``OPD_rms = (lambda / 2pi) * sigma_rad`` (the inverse of
    ``sigma_phi = (2pi/lambda) * sigma_WFE``; research/07 C.1).
    """
    return float(wavelength_m / (2.0 * np.pi) * sigma_rad)


def meters_to_rad(sigma_m: float, wavelength_m: float) -> float:
    """Convert an RMS optical-path in metres to RMS phase in radians
    (``sigma_phi = (2pi/lambda) * sigma_m``)."""
    return float(2.0 * np.pi / wavelength_m * sigma_m)


# ============================================================================
# 2. Strehl ratio (extended Marechal approximation)
# ============================================================================

def strehl_marechal(phase_or_sigma: Union[np.ndarray, float],
                    mask: Optional[np.ndarray] = None) -> float:
    """Strehl ratio via the (extended) Marechal approximation
    ``S = exp(-sigma_phi_res^2)`` (research/07 C.1).

    ``phase_or_sigma`` may be either

    * a **residual phase map** (array): its piston-removed RMS ``sigma_phi``
      (radians) is computed and ``S = exp(-sigma_phi^2)`` returned; or
    * a scalar **residual variance** ``sigma_phi_res^2`` (radians^2): used
      directly as the exponent (``S = exp(-variance)``).

    Valid for ``sigma_phi <~ 0.5`` rad (S >~ 0.6 within ~10 %); the formula is
    still returned outside that range as a (decreasingly accurate) estimate.
    The result is clipped to ``[0, 1]``.
    """
    arr = np.asarray(phase_or_sigma, dtype=np.float64)
    if arr.ndim == 0:
        # Scalar: interpret as residual variance (sigma_phi^2) in rad^2.
        variance = float(arr)
        if variance < 0.0:
            raise ValueError(
                f"residual variance must be non-negative, got {variance}")
    else:
        a = _as_masked_flat(arr, mask)
        if a.size == 0:
            return 1.0
        a = _remove_piston(a)
        variance = float(np.mean(a * a))
    S = float(np.exp(-variance))
    return float(np.clip(S, 0.0, 1.0))


# ============================================================================
# 3. Residual variance
# ============================================================================

def residual_variance(true_phase: np.ndarray, recon_phase: np.ndarray,
                      mask: Optional[np.ndarray] = None) -> float:
    """Variance of the residual ``true_phase - recon_phase`` over the pupil,
    after piston removal (research/07 C.1).

    This is the quantity the Marechal Strehl depends on
    (``S = exp(-residual_variance)`` when the residual is in radians).
    ``Var`` here is the mean of the squared, piston-removed residual (population
    variance / mean-square about the mean), so ``residual_variance(phi, 0)``
    equals the population variance of ``phi``.
    """
    res = np.asarray(true_phase, dtype=np.float64) - np.asarray(
        recon_phase, dtype=np.float64)
    a = _as_masked_flat(res, mask)
    if a.size == 0:
        return 0.0
    a = _remove_piston(a)
    return float(np.mean(a * a))


# ============================================================================
# 4. Phase correlation (Pearson)
# ============================================================================

def phase_correlation(W_recon: np.ndarray, W_true: np.ndarray,
                      mask: Optional[np.ndarray] = None) -> float:
    """Pearson cross-correlation between two phase maps over the valid pupil
    (research/07 C.1).

    ``rho = cov(W_recon, W_true) / (sigma_recon * sigma_true)`` in ``[-1, 1]``:
    ``+1`` for identical maps (up to a positive scale + piston), ``-1`` for a
    map vs its negative, ``~0`` for uncorrelated maps.  If either map is
    constant over the pupil (zero variance) the correlation is undefined and
    ``0.0`` is returned.
    """
    a = np.asarray(W_recon, dtype=np.float64)
    b = np.asarray(W_true, dtype=np.float64)
    # Apply the mask jointly so both vectors index the same pixels.  Build a
    # combined finite+mask selector to keep the two arrays aligned.
    af = a.ravel()
    bf = b.ravel()
    if af.shape != bf.shape:
        raise ValueError(
            f"phase maps must have the same shape; got {a.shape} and {b.shape}")
    if mask is not None:
        m = np.asarray(mask, dtype=bool).ravel()
        if m.shape != af.shape:
            raise ValueError("mask shape incompatible with phase-map shape")
        af = af[m]
        bf = bf[m]
    good = np.isfinite(af) & np.isfinite(bf)
    af = af[good]
    bf = bf[good]
    if af.size < 2:
        return 0.0
    # Reference scales BEFORE centring, to detect a (near-)constant map whose
    # post-centring norm is only floating-point dust.
    scale_a = float(np.max(np.abs(af))) if af.size else 0.0
    scale_b = float(np.max(np.abs(bf))) if bf.size else 0.0
    af = af - af.mean()
    bf = bf - bf.mean()
    na = float(np.sqrt(np.sum(af * af)))
    nb = float(np.sqrt(np.sum(bf * bf)))
    # A map is effectively constant (zero variance) if its centred norm is
    # negligible relative to its magnitude * sqrt(N); then correlation is
    # undefined and we return 0.0.
    eps = np.finfo(np.float64).eps
    tol_a = 1e3 * eps * scale_a * np.sqrt(af.size)
    tol_b = 1e3 * eps * scale_b * np.sqrt(bf.size)
    if na <= tol_a or nb <= tol_b:
        return 0.0
    rho = float(np.dot(af, bf) / (na * nb))
    # Guard tiny floating overshoot beyond +-1.
    return float(np.clip(rho, -1.0, 1.0))


# ============================================================================
# 5. r0 / tau0 recovery error
# ============================================================================

def r0_recovery_error(r0_est: float, r0_true: float,
                      percent: bool = True, signed: bool = False) -> float:
    """Relative r0 recovery error (research/07 C.1).

    By default returns the **percent absolute** relative error
    ``100 * |r0_est - r0_true| / r0_true``.  With ``signed=True`` the sign is
    kept (``(r0_est - r0_true) / r0_true``), so an over-estimate is positive --
    e.g. ``r0_recovery_error(1.05, 1.0)`` is ``5.0`` (%) and
    ``r0_recovery_error(1.05, 1.0, signed=True)`` is ``+5.0`` (%).  With
    ``percent=False`` the bare fraction (not x100) is returned.
    """
    if r0_true == 0.0:
        raise ValueError("r0_true must be non-zero to form a relative error")
    rel = (float(r0_est) - float(r0_true)) / float(r0_true)
    if not signed:
        rel = abs(rel)
    return float(rel * 100.0) if percent else float(rel)


def tau0_recovery_error(tau0_est: float, tau0_true: float,
                        percent: bool = True, signed: bool = False) -> float:
    """Relative tau0 recovery error (same convention as
    :func:`r0_recovery_error`)."""
    if tau0_true == 0.0:
        raise ValueError("tau0_true must be non-zero to form a relative error")
    rel = (float(tau0_est) - float(tau0_true)) / float(tau0_true)
    if not signed:
        rel = abs(rel)
    return float(rel * 100.0) if percent else float(rel)


# ============================================================================
# 6. DM-corrected residual
# ============================================================================

def residual_after_dm(true_phase_nodes: np.ndarray, H: np.ndarray,
                      commands: np.ndarray,
                      mask: Optional[np.ndarray] = None,
                      reflection_factor: float = -0.5
                      ) -> Tuple[np.ndarray, float]:
    """Residual wavefront after applying a DM command vector, with the
    reflection convention baked in (research/07 C.1, C.2; research/05 S9).

    The DM mechanical surface is ``s = H @ commands`` (``H`` = influence matrix,
    columns = per-actuator influence functions sampled at the same nodes as
    ``true_phase_nodes``).  A reflective DM imprints the wavefront correction

        W_dm = (1 / reflection_factor) * s = -2 * (H @ commands)

    for the default ``reflection_factor = -0.5`` (the same factor baked into
    ``aokit.dm.build_command_matrix``: commands are fitted so ``H @ commands``
    approximates the conjugate target ``reflection_factor * phi = -phi/2``).
    The corrected residual is

        W_res = true_phase_nodes - W_dm = true_phase_nodes + 2 * (H @ commands).

    Consequently, if ``commands == reflection_factor * true_phase`` and ``H`` is
    perfect (identity / no fitting error) the residual is zero; any departure is
    the **inter-actuator-coupling fitting error** ``sigma_fit`` this metric
    quantifies.

    Returns ``(W_res, rms)`` where ``rms`` is the piston-removed RMS of the
    residual over the (optional) pupil ``mask``, in radians.
    """
    true_nodes = np.asarray(true_phase_nodes, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    commands = np.asarray(commands, dtype=np.float64)

    surface = H @ commands                      # mechanical mirror surface
    if reflection_factor == 0.0:
        raise ValueError("reflection_factor must be non-zero")
    W_dm = surface / reflection_factor          # = -2 * surface (default)
    W_res = true_nodes - W_dm
    rms = rms_wfe(W_res, mask=mask, remove_piston=True)
    return W_res, rms


def dm_residual(W_true: np.ndarray, dm_surface: np.ndarray,
                mask: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """Residual after DM correction from an already-realised correction map
    (research/07 C.1).

    Here ``dm_surface`` is the **wavefront correction** ``W_dm`` the DM applies
    (already including the reflection factor, i.e. in the same units/sign as
    ``W_true``), so the residual is the plain difference
    ``W_res = W_true - dm_surface``.  Returns ``(rms_residual, strehl)`` with the
    piston-removed RMS (radians) and the Marechal Strehl
    ``exp(-sigma_res^2)``.  Use :func:`residual_after_dm` instead when you have
    raw actuator commands + the influence matrix ``H``.
    """
    W_res = np.asarray(W_true, dtype=np.float64) - np.asarray(
        dm_surface, dtype=np.float64)
    rms = rms_wfe(W_res, mask=mask, remove_piston=True)
    strehl = strehl_marechal(rms * rms)         # exponent = residual variance
    return float(rms), float(strehl)


# ============================================================================
# 7. Zernike-coefficient decomposition error
# ============================================================================

def zernike_decomp_error(true_coeffs: np.ndarray, est_coeffs: np.ndarray
                         ) -> dict:
    """Per-mode and total Zernike-coefficient recovery error (research/07 C.3).

    ``true_coeffs`` / ``est_coeffs`` are Noll-ordered coefficient vectors (same
    length).  Returns a dict with

    * ``per_mode``     -- signed error ``est - true`` per mode,
    * ``abs_per_mode`` -- ``|est - true|`` per mode,
    * ``total_rms``    -- RMS coefficient error ``sqrt(mean((est-true)^2))``,
    * ``total_l2``     -- Euclidean norm ``||est - true||_2`` (= total residual
                          wavefront RMS, since the Noll basis is orthonormal so
                          coefficient L2 == wavefront RMS),
    * ``max_abs``      -- worst single-mode absolute error,
    * ``rel_l2``       -- ``||est-true||_2 / ||true||_2`` (0 if ``true`` is 0).
    """
    t = np.asarray(true_coeffs, dtype=np.float64).ravel()
    e = np.asarray(est_coeffs, dtype=np.float64).ravel()
    if t.shape != e.shape:
        raise ValueError(
            f"coefficient vectors must match in length; got {t.shape} and "
            f"{e.shape}")
    diff = e - t
    n = diff.size
    total_l2 = float(np.sqrt(np.sum(diff * diff)))
    norm_true = float(np.sqrt(np.sum(t * t)))
    return {
        "per_mode": diff,
        "abs_per_mode": np.abs(diff),
        "total_rms": float(np.sqrt(np.mean(diff * diff))) if n else 0.0,
        "total_l2": total_l2,
        "max_abs": float(np.max(np.abs(diff))) if n else 0.0,
        "rel_l2": float(total_l2 / norm_true) if norm_true > 0.0 else 0.0,
    }


# ============================================================================
# 8. Error-budget summary
# ============================================================================

def error_budget(W_recon: np.ndarray, W_true: np.ndarray,
                 mask: Optional[np.ndarray] = None,
                 wavelength_m: Optional[float] = None,
                 r0_est: Optional[float] = None, r0_true: Optional[float] = None,
                 tau0_est: Optional[float] = None,
                 tau0_true: Optional[float] = None) -> dict:
    """Summary error-budget dict combining the core metrics (research/07 C.2).

    Works on a **single frame** (2-D ``W_recon``/``W_true``) or a **time-series**
    (stack with a leading frame axis, or any pair of equal-shaped arrays): the
    residual statistics are computed over all valid samples jointly and the
    returned values are ensemble means.

    Always present keys: ``rms_wfe_rad``, ``residual_variance_rad2``,
    ``strehl_marechal``, ``phase_correlation``, ``n_pix``.  When
    ``wavelength_m`` is given, ``rms_wfe_m`` (metres of OPD) is added.  When the
    matching ``*_est``/``*_true`` pair is given, ``r0_recovery_error_pct`` and/or
    ``tau0_recovery_error_pct`` are added.  All values are finite floats.
    """
    recon = np.asarray(W_recon, dtype=np.float64)
    true = np.asarray(W_true, dtype=np.float64)
    if recon.shape != true.shape:
        raise ValueError(
            f"W_recon and W_true must match in shape; got {recon.shape} and "
            f"{true.shape}")

    # Mask handling for stacked time-series: broadcast a per-frame mask across
    # the leading axis if needed.
    eff_mask = mask
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape != recon.shape:
            if recon.ndim == m.ndim + 1 and recon.shape[1:] == m.shape:
                eff_mask = np.broadcast_to(m, recon.shape)
            else:
                raise ValueError(
                    f"mask shape {m.shape} incompatible with data shape "
                    f"{recon.shape}")

    rms = rms_wfe(recon, true, mask=eff_mask, remove_piston=True)
    var = residual_variance(true, recon, mask=eff_mask)
    strehl = strehl_marechal(var)
    rho = phase_correlation(recon, true, mask=eff_mask)
    n_pix = int(_as_masked_flat(recon - true, eff_mask).size)

    out = {
        "rms_wfe_rad": float(rms),
        "residual_variance_rad2": float(var),
        "strehl_marechal": float(strehl),
        "phase_correlation": float(rho),
        "n_pix": n_pix,
    }
    if wavelength_m is not None:
        out["rms_wfe_m"] = rad_to_meters(rms, float(wavelength_m))
    if r0_est is not None and r0_true is not None:
        out["r0_recovery_error_pct"] = r0_recovery_error(r0_est, r0_true)
    if tau0_est is not None and tau0_true is not None:
        out["tau0_recovery_error_pct"] = tau0_recovery_error(tau0_est, tau0_true)
    return out


# ============================================================================
# 9. Reconstructor self-consistency & C/Python parity
# ============================================================================

def reconstructor_self_consistency(slopes_meas: np.ndarray,
                                   W_recon: np.ndarray, Gamma: np.ndarray
                                   ) -> float:
    """Round-trip residual ``||Gamma @ phi_recon - slopes_meas||`` (research/07
    C.2).  Re-applying the (Fried/Hudgin) slope operator ``Gamma`` to the
    reconstructed phase must reproduce the measured slopes; the returned L2 norm
    of the discrepancy should be ~0 for a self-consistent reconstructor.
    """
    Gamma = np.asarray(Gamma, dtype=np.float64)
    phi = np.asarray(W_recon, dtype=np.float64).ravel()
    s = np.asarray(slopes_meas, dtype=np.float64).ravel()
    pred = Gamma @ phi
    return float(np.linalg.norm(pred - s))


def cpython_parity(slopes_c: np.ndarray, slopes_py: np.ndarray) -> float:
    """Max absolute difference between C-core and Python-reference slopes for the
    same frame (research/07 C.2) -- guards the fast path.  Returns
    ``max |slopes_c - slopes_py|``.
    """
    c = np.asarray(slopes_c, dtype=np.float64).ravel()
    p = np.asarray(slopes_py, dtype=np.float64).ravel()
    if c.shape != p.shape:
        raise ValueError(
            f"slope vectors must match in length; got {c.shape} and {p.shape}")
    if c.size == 0:
        return 0.0
    return float(np.max(np.abs(c - p)))


def noll_variance_check(coeffs_ts: np.ndarray, D_m: float, r0_true: float
                        ) -> np.ndarray:
    """Monte-Carlo check: ensemble Zernike variances should follow Noll
    ``c_j (D/r0)^(5/3)``; returns per-mode ratio (measured/expected)
    (research/07 C.4).

    ``coeffs_ts`` is ``(n_frames, n_modes)`` Noll-ordered (column ``k`` is mode
    ``j = k + 1``, i.e. starting at piston).  The measured per-mode variance
    across frames is divided by the theoretical ``noll_variance(j) *
    (D/r0)^(5/3)``; a well-calibrated screen+pipeline gives ratios ~1 (piston is
    skipped -> NaN for j=1, which carries no Kolmogorov power).
    """
    from .zernike import noll_variance

    C = np.asarray(coeffs_ts, dtype=np.float64)
    if C.ndim != 2:
        raise ValueError(
            f"coeffs_ts must be (n_frames, n_modes); got shape {C.shape}")
    n_modes = C.shape[1]
    measured = C.var(axis=0)                    # population variance per mode
    scale = (float(D_m) / float(r0_true)) ** (5.0 / 3.0)
    ratio = np.full(n_modes, np.nan, dtype=np.float64)
    for k in range(n_modes):
        j = k + 1
        expected = noll_variance(j) * scale
        if expected > 0.0:
            ratio[k] = measured[k] / expected
    return ratio
