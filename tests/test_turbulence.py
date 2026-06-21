"""Unit tests for aokit.turbulence (>=7 r0 + >=6 tau0 estimators + combiner).

Validation strategy (research/04 C.3): construct synthetic data with **injected
ground truth** directly -- no dependence on datagen.py.

* r0: draw ensembles of Zernike-coefficient vectors from the Noll-Kolmogorov
  covariance scaled by ``(D/r0)^(5/3)`` for a KNOWN r0; render Kolmogorov phase
  maps on a pupil from those modes; build slope series from the G-tilt variance
  relation.  Each estimator must recover the injected r0.
* tau0: build an AR(1) (frozen-flow-like) series with a known 1/e correlation
  time; the autocorrelation / structure-function estimators must recover it.
  Greenwood/Tyler are exact numeric closures.
"""
import numpy as np
import pytest

from aokit import turbulence as tb
from aokit.zernike import noll_covariance, zernike_array
from aokit.config import from_dict


# ----------------------------------------------------------------------------
# Synthetic-data helpers (injected ground truth)
# ----------------------------------------------------------------------------

def _kolmogorov_cov(j_list, D_m, r0_m):
    """Noll-Kolmogorov coefficient covariance ``<a_j a_j'>`` scaled by
    ``(D/r0)^(5/3)`` for the given Noll indices (a (J, J) SPD matrix)."""
    n = len(j_list)
    cov = np.empty((n, n))
    for i, ji in enumerate(j_list):
        for k, jk in enumerate(j_list):
            cov[i, k] = noll_covariance(int(ji), int(jk))
    cov *= (D_m / r0_m) ** (5.0 / 3.0)
    return 0.5 * (cov + cov.T)


def _cov_sqrt(cov):
    """Symmetric matrix square-root (eigh, clipped to PSD)."""
    w, V = np.linalg.eigh(cov)
    w = np.clip(w, 0.0, None)
    return V @ np.diag(np.sqrt(w))


def _draw_coeffs(j_list, D_m, r0_m, n_samples, rng):
    """``(n_samples, J)`` i.i.d. Kolmogorov Zernike coefficient vectors."""
    cov = _kolmogorov_cov(j_list, D_m, r0_m)
    Lh = _cov_sqrt(cov)
    z = rng.standard_normal((len(j_list), n_samples))
    return (Lh @ z).T


def _ar1_coeffs(j_list, D_m, r0_m, n_frames, rho, rng):
    """Kolmogorov-correlated-in-space, AR(1)-correlated-in-time coefficient
    series ``(n_frames, J)`` with per-step lag-1 correlation ``rho`` (so the
    normalised temporal ACF is ``rho^k = exp(-k dt / tau_c)``)."""
    cov = _kolmogorov_cov(j_list, D_m, r0_m)
    Lh = _cov_sqrt(cov)
    n = len(j_list)
    white = rng.standard_normal((n_frames, n))
    ar = np.empty_like(white)
    ar[0] = white[0]
    s = np.sqrt(1.0 - rho ** 2)
    for t in range(1, n_frames):
        ar[t] = rho * ar[t - 1] + s * white[t]
    return (Lh @ ar.T).T


def _render_phase_maps(coeffs, j_max, n_grid):
    """Render modal coefficients ``(F, J)`` (Noll j = 2..j_max) into pupil phase
    maps ``(F, n_grid, n_grid)`` with NaN outside the unit disc."""
    Zstack = zernike_array(j_max, n_grid)            # (J, n_pix) modes 2..j_max
    lin = np.linspace(-1.0, 1.0, n_grid)
    xx, yy = np.meshgrid(lin, lin)
    mask = (xx ** 2 + yy ** 2) <= 1.0
    F = coeffs.shape[0]
    flat = np.full((F, n_grid * n_grid), np.nan)
    flat[:, mask.ravel()] = coeffs @ Zstack
    return flat.reshape(F, n_grid, n_grid), mask


def _make_cfg(D_m=1.0, r0_m=0.15, tau0_s=0.0047, wind=10.0, d_sub=0.1):
    return from_dict({
        "schema_version": 1,
        "camera": {"pixel_size_m": 5.5e-6, "frame_w": 256, "frame_h": 256,
                   "bit_depth": 8},
        "mla": {"n_lenslets_x": 10, "n_lenslets_y": 10, "pitch_m": d_sub,
                "focal_length_m": 5.2e-3},
        "pupil": {"diameter_m": D_m, "center_x_px": 128.0, "center_y_px": 128.0},
        "wavelength_m": 6.33e-7,
        "dm": {"n_act_x": 11, "n_act_y": 11, "pitch_m": d_sub,
               "coupling_coeff": 0.15, "stroke_max_m": 3.5e-6},
        "geometry": {"type": "fried"},
        "cadence": {"dt_s": 2.0e-3},
        "ground_truth": {"r0_m": r0_m, "tau0_s": tau0_s, "wind_speed_mps": wind},
    })


# ============================ r0 ESTIMATORS =============================

def test_r0_zernike_variance_recovers_injected():
    """R1: from an ensemble of screens at known r0, fitted r0 is within a few %
    (research/04 R1)."""
    rng = np.random.default_rng(42)
    D, r0 = 1.0, 0.15
    j_list = np.arange(2, 36)
    coeffs = _draw_coeffs(j_list, D, r0, n_samples=40000, rng=rng)
    r0_est = tb.r0_from_zernike_variance(coeffs, D, modes=(4, 30))
    assert abs(r0_est - r0) / r0 < 0.05


def test_r0_zernike_variance_excludes_tiptilt():
    """R1 must ignore tip/tilt even if the window requests them (they are
    vibration-contaminated; research/04 S3.3). Injecting huge extra tip/tilt
    does not move r0."""
    rng = np.random.default_rng(7)
    D, r0 = 1.0, 0.15
    j_list = np.arange(2, 36)
    coeffs = _draw_coeffs(j_list, D, r0, n_samples=30000, rng=rng)
    # Blow up tip (col 0) and tilt (col 1) by 10x:
    contaminated = coeffs.copy()
    contaminated[:, 0:2] *= 10.0
    r0_clean = tb.r0_from_zernike_variance(coeffs, D, modes=(4, 30))
    r0_dirty = tb.r0_from_zernike_variance(contaminated, D, modes=(2, 30))
    assert abs(r0_dirty - r0_clean) / r0_clean < 1e-6


def test_r0_zernike_variance_noise_subtraction():
    """R1: adding white per-mode measurement noise and supplying the noise
    variance recovers the unbiased r0 (research/04 S3.3 bias removal)."""
    rng = np.random.default_rng(3)
    D, r0 = 1.0, 0.15
    j_list = np.arange(2, 36)
    coeffs = _draw_coeffs(j_list, D, r0, n_samples=40000, rng=rng)
    sigma_n = 0.02
    noisy = coeffs + rng.normal(0.0, sigma_n, coeffs.shape)
    nvec = np.full(coeffs.shape[1], sigma_n ** 2)
    r0_est = tb.r0_from_zernike_variance(noisy, D, modes=(4, 30), noise_var=nvec)
    assert abs(r0_est - r0) / r0 < 0.07


def test_r0_slope_variance_independent_of_reconstruction():
    """R2: slope-variance r0 recovers the injected value without reconstruction
    (research/04 R2)."""
    rng = np.random.default_rng(11)
    lam, d, r0 = 6.33e-7, 0.1, 0.15
    alpha2 = tb.SLOPE_GTILT_CONST * lam ** 2 * r0 ** (-5.0 / 3.0) * d ** (-1.0 / 3.0)
    M, T = 60, 30000
    sx = rng.normal(0.0, np.sqrt(alpha2), (T, M))
    sy = rng.normal(0.0, np.sqrt(alpha2), (T, M))
    slopes = np.concatenate([sx, sy], axis=1)
    r0_est = tb.r0_from_slope_variance(slopes, lam, d)
    assert abs(r0_est - r0) / r0 < 0.03


def test_r0_slope_variance_noise_subtraction():
    """R2: subtracting the centroid-noise variance de-biases r0."""
    rng = np.random.default_rng(12)
    lam, d, r0 = 6.33e-7, 0.1, 0.15
    alpha2 = tb.SLOPE_GTILT_CONST * lam ** 2 * r0 ** (-5.0 / 3.0) * d ** (-1.0 / 3.0)
    M, T = 60, 30000
    noise_var = 0.2 * alpha2
    sx = rng.normal(0.0, np.sqrt(alpha2 + noise_var), (T, M))
    sy = rng.normal(0.0, np.sqrt(alpha2 + noise_var), (T, M))
    slopes = np.concatenate([sx, sy], axis=1)
    r0_biased = tb.r0_from_slope_variance(slopes, lam, d)
    r0_corr = tb.r0_from_slope_variance(slopes, lam, d, noise_var=noise_var)
    # noise inflates variance -> biases r0 LOW; correction recovers it.
    assert r0_biased < r0_corr
    assert abs(r0_corr - r0) / r0 < 0.04


def test_r0_dimm_recovers_and_is_vibration_immune():
    """R3: DIMM r0 recovers the injected value AND is unaffected by an injected
    common-mode tip/tilt jitter (research/04 R3)."""
    rng = np.random.default_rng(21)
    lam, d, r0 = 6.33e-7, 0.1, 0.15
    B = 2.0 * d
    b = B / d
    k_l, k_t = tb._dimm_response(b)
    var_l = k_l * lam ** 2 * r0 ** (-5.0 / 3.0) * d ** (-1.0 / 3.0)
    var_t = k_t * lam ** 2 * r0 ** (-5.0 / 3.0) * d ** (-1.0 / 3.0)
    T = 40000
    dl = rng.normal(0.0, np.sqrt(var_l), T)
    dtv = rng.normal(0.0, np.sqrt(var_t), T)

    # Clean differential-only pair.
    pair_clean = np.column_stack([0.5 * dl, 0.5 * dtv, -0.5 * dl, -0.5 * dtv])
    r0_clean = tb.r0_from_dimm(pair_clean, B, d, lam)
    assert abs(r0_clean - r0) / r0 < 0.05

    # Add a large common-mode jitter (>> differential motion).
    cx = rng.normal(0.0, 1e-4, T)
    cy = rng.normal(0.0, 1e-4, T)
    pair_jit = np.column_stack([cx + 0.5 * dl, cy + 0.5 * dtv,
                                cx - 0.5 * dl, cy - 0.5 * dtv])
    r0_jit = tb.r0_from_dimm(pair_jit, B, d, lam)
    assert abs(r0_jit - r0_clean) < 1e-9         # exactly cancels


def test_r0_total_variance_exact():
    """R4 (scalar): inverting sigma^2 = 1.0299 (D/r0)^5/3 recovers r0 exactly,
    both piston-removed and tip/tilt-removed (research/04 R4)."""
    D, r0 = 1.0, 0.15
    s2 = tb.KOLM_TOTAL_VAR * (D / r0) ** (5.0 / 3.0)
    assert abs(tb.r0_from_total_variance(s2, D) - r0) < 1e-9
    s2tt = tb.KOLM_TT_REMOVED_VAR * (D / r0) ** (5.0 / 3.0)
    assert abs(tb.r0_from_total_variance(s2tt, D, tt_removed=True) - r0) < 1e-9


def test_r0_phase_variance_recovers_injected():
    """R4 (maps): the spatial variance of TT-removed Kolmogorov phase maps
    inverts to the injected r0 (research/04 R4)."""
    rng = np.random.default_rng(31)
    D, r0 = 1.0, 0.15
    j_max = 120
    j_list = np.arange(2, j_max + 1)
    # Draw many frames, drop tip/tilt (cols 0,1) so the residual matches 0.134.
    coeffs = _draw_coeffs(j_list, D, r0, n_samples=400, rng=rng)
    coeffs[:, 0:2] = 0.0
    maps, mask = _render_phase_maps(coeffs, j_max, n_grid=64)
    # Pull out only the in-pupil pixels per frame for a clean spatial variance.
    F = maps.shape[0]
    pix = maps.reshape(F, -1)[:, mask.ravel()]
    r0_est = tb.r0_from_phase_variance(pix, D, tt_removed=True)
    # Finite mode count truncates high-order content -> small underestimate of
    # variance -> r0 slightly high; require order-of-magnitude + ~15%.
    assert abs(r0_est - r0) / r0 < 0.15


def test_r0_structure_function_recovers_and_slope_is_5_3():
    """R5: structure-function fit recovers the injected r0 and its log-log slope
    is ~5/3, validating Kolmogorov (research/04 R5)."""
    rng = np.random.default_rng(55)
    D, r0 = 1.0, 0.15
    j_max = 300
    j_list = np.arange(2, j_max + 1)
    coeffs = _draw_coeffs(j_list, D, r0, n_samples=40, rng=rng)
    n_grid = 80
    maps, mask = _render_phase_maps(coeffs, j_max, n_grid)
    pitch = D / n_grid
    r0_est, slope = tb.r0_from_structure_function(
        maps, pitch, r_range=(4 * pitch, 0.45 * D), pupil_mask=mask)
    assert abs(r0_est - r0) / r0 < 0.10
    assert abs(slope - 5.0 / 3.0) < 0.25         # ~1.667


def test_r0_vonkarman_matches_kolmogorov_limit():
    """R6: the joint (r0, L0) fit reduces to the Kolmogorov r0 (and returns a
    large/inf L0) for pure-Kolmogorov input (research/04 R6)."""
    rng = np.random.default_rng(61)
    D, r0 = 1.0, 0.15
    j_list = np.arange(2, 36)
    coeffs = _draw_coeffs(j_list, D, r0, n_samples=40000, rng=rng)
    r0_vk, L0 = tb.r0_l0_from_vonkarman(coeffs, D, modes=(4, 30))
    assert abs(r0_vk - r0) / r0 < 0.05
    assert L0 > D                                 # weak/large outer scale
    # Convenience scalar wrapper agrees.
    assert abs(tb.r0_from_vonkarman(coeffs, D, modes=(4, 30)) - r0_vk) < 1e-9


def test_r0_seeing_exact():
    """R7: r0 = 0.98 lambda / epsilon is an exact numeric inversion
    (research/04 R7)."""
    lam, r0 = 6.33e-7, 0.15
    eps = tb.SEEING_CONST * lam / r0
    assert abs(tb.r0_from_seeing(eps, lam) - r0) < 1e-12


def test_r0_estimators_agree_within_spread():
    """R1..R7 agree within their spread on a synthetic dataset; the combiner's
    median is close to the injected r0 (research/04 S5)."""
    rng = np.random.default_rng(99)
    D, r0, lam, d = 1.0, 0.15, 6.33e-7, 0.1
    j_list = np.arange(2, 36)
    coeffs = _draw_coeffs(j_list, D, r0, n_samples=40000, rng=rng)
    alpha2 = tb.SLOPE_GTILT_CONST * lam ** 2 * r0 ** (-5.0 / 3.0) * d ** (-1.0 / 3.0)
    slopes = rng.normal(0.0, np.sqrt(alpha2), (20000, 2 * 50))

    vals = {
        "R1": tb.r0_from_zernike_variance(coeffs, D, modes=(4, 30)),
        "R2": tb.r0_from_slope_variance(slopes, lam, d),
        "R4": tb.r0_from_total_variance(
            tb.KOLM_TOTAL_VAR * (D / r0) ** (5.0 / 3.0), D),
        "R6": tb.r0_from_vonkarman(coeffs, D, modes=(4, 30)),
        "R7": tb.r0_from_seeing(tb.SEEING_CONST * lam / r0, lam),
    }
    median, spread = tb.combine_estimates(vals)
    assert abs(median - r0) / r0 < 0.05
    assert np.isfinite(spread) and spread < 0.05 * r0


# ============================ tau0 ESTIMATORS ============================

def test_tau0_autocorrelation_recovers_injected():
    """T1: 1/e of the temporal autocorrelation recovers the injected tau0 from a
    frozen-flow / AR(1) series (research/04 T1)."""
    rng = np.random.default_rng(101)
    dt = 2.0e-3
    tau_c = 10.0e-3                               # injected 1/e coherence time
    rho = np.exp(-dt / tau_c)
    T = 200000
    x = np.empty(T)
    x[0] = rng.standard_normal()
    s = np.sqrt(1.0 - rho ** 2)
    for t in range(1, T):
        x[t] = rho * x[t - 1] + s * rng.standard_normal()
    tau0_est = tb.tau0_from_autocorr(x, dt)
    assert abs(tau0_est - tau_c) / tau_c < 0.10
    # the documented alias gives the same number
    assert tb.tau0_from_autocorrelation(x, dt) == tau0_est


def test_tau0_autocorr_isolates_noise_jump():
    """T1: white measurement noise sits only at tau=0, so the 1/e time is
    unchanged by added noise (the noise-isolation dual use; research/04 T1)."""
    rng = np.random.default_rng(102)
    dt = 2.0e-3
    tau_c = 12.0e-3
    rho = np.exp(-dt / tau_c)
    T = 150000
    x = np.empty(T)
    x[0] = rng.standard_normal()
    s = np.sqrt(1.0 - rho ** 2)
    for t in range(1, T):
        x[t] = rho * x[t - 1] + s * rng.standard_normal()
    noisy = x + rng.normal(0.0, 0.5, T)           # big white noise
    tau_clean = tb.tau0_from_autocorr(x, dt)
    tau_noisy = tb.tau0_from_autocorr(noisy, dt)
    assert abs(tau_noisy - tau_clean) / tau_clean < 0.10


def test_tau0_structure_function_recovers_injected():
    """T4: the temporal structure-function time tracks the injected coherence
    time (up to the documented t0 = 0.66 tau0 convention; research/04 T4)."""
    rng = np.random.default_rng(103)
    dt = 2.0e-3
    tau_c = 10.0e-3
    rho = np.exp(-dt / tau_c)
    T = 200000
    x = np.empty(T)
    x[0] = rng.standard_normal()
    s = np.sqrt(1.0 - rho ** 2)
    for t in range(1, T):
        x[t] = rho * x[t - 1] + s * rng.standard_normal()
    tau0_t4 = tb.tau0_from_structure_function(x, dt)
    # T4 reports tau0 = t_e * 2^(3/5) where t_e is the e-fold (=tau_c here).
    expected = tau_c * tb.T0_TO_TAU0
    assert abs(tau0_t4 - expected) / expected < 0.10
    # And it is immune to the additive noise floor (cancels for tau>0).
    noisy = x + rng.normal(0.0, 0.5, T)
    assert abs(tb.tau0_from_structure_function(noisy, dt) - tau0_t4) / tau0_t4 < 0.1


def test_tau0_psd_orders_with_coherence_time():
    """T2: the PSD-knee timescale is finite and grows with the coherence time
    (slower turbulence -> lower knee -> larger tau0; research/04 T2)."""
    rng = np.random.default_rng(104)
    dt = 1.0e-3
    T = 100000

    def ar1(tau_c):
        rho = np.exp(-dt / tau_c)
        x = np.empty(T)
        x[0] = rng.standard_normal()
        sc = np.sqrt(1.0 - rho ** 2)
        for t in range(1, T):
            x[t] = rho * x[t - 1] + sc * rng.standard_normal()
        return x

    tau_fast, tau_slow = 5.0e-3, 20.0e-3
    t2_fast, fc_fast = tb.tau0_from_psd(ar1(tau_fast), dt)
    t2_slow, fc_slow = tb.tau0_from_psd(ar1(tau_slow), dt)
    assert np.isfinite(t2_fast) and np.isfinite(t2_slow)
    assert t2_slow > t2_fast                      # slower turbulence -> larger tau0
    assert fc_slow < fc_fast                      # and a lower knee frequency


def test_greenwood_exact_and_closure():
    """T3: tau0 = 0.314 r0/v = 0.134/f_G with f_G = 0.426 v/r0 (exact numeric
    closure; research/04 T3, S5)."""
    r0, v = 0.15, 10.0
    tau0, f_g = tb.tau0_from_greenwood(r0, v)
    assert abs(tau0 - tb.GREENWOOD_TAU0_CONST * r0 / v) < 1e-15
    assert abs(f_g - tb.GREENWOOD_FREQ_CONST * v / r0) < 1e-12
    # 0.314 r0/v ~= 0.134 / f_G to ~1% (the two constants are consistent).
    assert abs(tau0 - tb.GREENWOOD_BRIDGE / f_g) / tau0 < 0.02


def test_tau0_frozen_flow_wind_matches_injected():
    """T5: spatio-temporal cross-correlation recovers the injected wind speed,
    hence tau0 = 0.314 r0/v (research/04 T5)."""
    pytest.importorskip("scipy")
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(201)
    N, T = 48, 25
    base = gaussian_filter(rng.standard_normal((220, 220)), 3.0)
    dx_true, dy_true = 3.0, 0.0                    # px / frame
    pitch, dt = 0.02, 2.0e-3
    off = 10
    maps = np.empty((T, N, N))
    for t in range(T):
        sx = int(round(dx_true * t))
        sy = int(round(dy_true * t))
        maps[t] = base[off + sy:off + sy + N, off + sx:off + sx + N]
    v, corr = tb.wind_from_frozen_flow(maps, dt, pitch, max_shift=8)
    v_true = np.hypot(dx_true, dy_true) * pitch / dt
    assert corr > 0.8
    assert abs(v - v_true) / v_true < 0.10
    # tau0 via the Greenwood bridge.
    r0 = 0.15
    tau0 = tb.tau0_from_taylor(maps, dt, pitch, r0)
    assert abs(tau0 - tb.GREENWOOD_TAU0_CONST * r0 / v_true) / \
        (tb.GREENWOOD_TAU0_CONST * r0 / v_true) < 0.12


def test_tyler_frequency_and_tracking_time():
    """T6: Tyler f_T = 0.368 v r0^-1/6 D^-5/6 (exact) and tau_T = 0.134/f_T
    (research/04 T6)."""
    r0, v, D = 0.15, 10.0, 1.0
    f_t = tb.tyler_frequency(r0, v, D)
    expect = tb.TYLER_CONST * v * r0 ** (-1.0 / 6.0) * D ** (-5.0 / 6.0)
    assert abs(f_t - expect) < 1e-12
    tau_t = tb.tau_tracking(r0, v, D)
    assert abs(tau_t - tb.GREENWOOD_BRIDGE / f_t) < 1e-15


# ============================== COMBINERS ===============================

def test_combine_estimates_median_and_spread():
    """combine_estimates returns a finite median + spread and ignores NaNs
    (research/04 S5)."""
    vals = {"a": 0.14, "b": 0.15, "c": 0.16, "bad": float("nan")}
    median, spread = tb.combine_estimates(vals)
    assert abs(median - 0.15) < 1e-9
    assert np.isfinite(spread) and spread > 0.0


def test_combine_r0_returns_full_crossval_dict():
    """combine_r0 returns per-method values plus mean/std/median for
    cross-validation (the task's combiner contract)."""
    vals = {"R1": 0.149, "R2": 0.146, "R4": 0.151}
    out = tb.combine_r0(vals)
    for k in ("R1", "R2", "R4", "mean", "std", "median"):
        assert k in out and np.isfinite(out[k])
    assert abs(out["median"] - 0.149) < 1e-9
    assert abs(out["mean"] - np.mean([0.149, 0.146, 0.151])) < 1e-12


def test_combine_tau0_returns_full_crossval_dict():
    """combine_tau0 mirrors combine_r0 (research/04 S5)."""
    vals = {"T1": 0.0045, "T3": 0.0047, "T4": 0.0044}
    out = tb.combine_tau0(vals)
    for k in ("T1", "T3", "T4", "mean", "std", "median"):
        assert k in out and np.isfinite(out[k])
    assert abs(out["median"] - 0.0045) < 1e-9


# ========================= TOP-LEVEL PIPELINE ===========================

def test_estimate_all_recovers_r0_and_populates_summary():
    """estimate_all runs every applicable estimator, medians them close to the
    injected r0, and fills the TurbulenceResult (research/04 S5 pipeline)."""
    rng = np.random.default_rng(303)
    D, r0, tau0, v, d = 1.0, 0.15, 0.0047, 10.0, 0.1
    cfg = _make_cfg(D_m=D, r0_m=r0, tau0_s=tau0, wind=v, d_sub=d)

    j_list = np.arange(2, 36)
    rho = np.exp(-cfg.dt_s / 0.010)
    coeffs = _ar1_coeffs(j_list, D, r0, n_frames=6000, rho=rho, rng=rng)
    alpha2 = tb.SLOPE_GTILT_CONST * cfg.wavelength_m ** 2 * \
        r0 ** (-5.0 / 3.0) * d ** (-1.0 / 3.0)
    slopes = rng.normal(0.0, np.sqrt(alpha2), (6000, 2 * 50))

    res = tb.estimate_all(slopes, coeffs, None, cfg, cfg.dt_s)
    # >=2 r0 estimators ran and the median is close to truth (median is robust
    # to the synthetic-slope DIMM outlier).
    assert len(res.r0_m) >= 3
    assert abs(res.r0_median - r0) / r0 < 0.10
    # tau0 estimators present and the median is finite & physical.
    assert len(res.tau0_s) >= 3
    assert np.isfinite(res.tau0_median) and 0.0 < res.tau0_median < 1.0
    # T1 (the assumption-free primary) recovers the injected ~10 ms.
    assert abs(res.tau0_s["T1_autocorr"] - 0.010) / 0.010 < 0.15
    # Summary fields populated.
    assert res.n_frames == 6000
    assert np.isfinite(res.r0_spread) and np.isfinite(res.tau0_spread)
    assert res.wind_speed_mps == v


def test_characterize_returns_summary_dict_schema():
    """characterize() returns the turbulence_summary.json-shaped dict
    (ARCHITECTURE.md S4.4)."""
    rng = np.random.default_rng(404)
    D, r0 = 1.0, 0.15
    cfg = _make_cfg(D_m=D, r0_m=r0)
    j_list = np.arange(2, 36)
    rho = np.exp(-cfg.dt_s / 0.010)
    coeffs = _ar1_coeffs(j_list, D, r0, n_frames=4000, rho=rho, rng=rng)
    alpha2 = tb.SLOPE_GTILT_CONST * cfg.wavelength_m ** 2 * \
        r0 ** (-5.0 / 3.0) * cfg.mla.pitch_m ** (-1.0 / 3.0)
    slopes = rng.normal(0.0, np.sqrt(alpha2), (4000, 2 * 50))

    summary = tb.characterize(slopes, coeffs, cfg.dt_s, cfg)
    assert set(summary.keys()) >= {
        "r0_m", "L0_m", "tau0_s", "wind_speed_mps", "f_greenwood_hz",
        "seeing_arcsec", "strehl_marechal", "n_frames", "dt_s", "notes"}
    assert set(summary["r0_m"].keys()) == {"median", "spread", "estimators"}
    assert set(summary["tau0_s"].keys()) == {"median", "spread", "estimators"}
    assert abs(summary["r0_m"]["median"] - r0) / r0 < 0.10
    # JSON-serialisable.
    import json
    json.dumps(summary)


def test_estimate_all_with_phase_maps_runs_struct_fn_and_wind():
    """estimate_all also exercises the phase-map domain (R4 maps, R5) and the
    frozen-flow wind path when phase maps are supplied."""
    rng = np.random.default_rng(505)
    D, r0 = 1.0, 0.15
    cfg = _make_cfg(D_m=D, r0_m=r0)
    j_max = 120
    j_list = np.arange(2, j_max + 1)
    coeffs = _draw_coeffs(j_list, D, r0, n_samples=40, rng=rng)
    coeffs[:, 0:2] = 0.0
    maps, _mask = _render_phase_maps(coeffs, j_max, n_grid=48)
    # NaN outside pupil is fine; R5 masks it. Provide a small modal series too.
    res = tb.estimate_all(None, coeffs, maps, cfg, cfg.dt_s)
    assert "R5_struct_fn" in res.r0_m
    assert np.isfinite(res.r0_m["R5_struct_fn"])
    assert np.isfinite(res.r0_median)
