"""Unit tests for aokit.validation (correctness & quality metrics).

Covers the research/07 PART C metrics: RMS WFE, Strehl (Marechal), residual
variance, phase correlation, r0/tau0 recovery error, DM-corrected residual
(with the -1/2 reflection convention), Zernike-coefficient error and the
error-budget summary.  Conventions asserted here (research/07 C.1-C.4):

  * RMS / variance / Strehl remove piston first (piston is unobservable).
  * Strehl is the extended Marechal ``S = exp(-sigma_phi_res^2)``.
  * phase_correlation is Pearson over the pupil: +1 identical, -1 negated, ~0
    uncorrelated.
  * residual_after_dm: wavefront correction is ``-2 * (H @ commands)`` (reflective
    DM, -1/2 reflection factor), so commands == -true/2 with H = I -> residual 0.
  * r0/tau0 recovery error is a relative error in percent.
"""
import numpy as np
import pytest

import aokit.validation as V


# --------------------------------------------------------------------------
# 1. Zero phase -> perfect metrics
# --------------------------------------------------------------------------

def test_zero_phase_rms_and_strehl():
    """Zero residual phase -> RMS WFE 0 and Strehl exactly 1.0 (research/07 C.1)."""
    phase = np.zeros((32, 32))
    assert V.rms_wfe(phase) == 0.0
    assert V.strehl_marechal(phase) == 1.0
    # Two-map form: identical maps -> zero error, Strehl 1.
    assert V.rms_wfe(phase, phase) == 0.0


def test_zero_residual_variance_and_constant_map_correlation():
    """Constant maps: residual variance 0; correlation against a *varying* map
    (zero variance on one side) is undefined -> 0.0 by convention."""
    phase = np.full((16, 16), 3.7)
    assert V.residual_variance(phase, phase) == 0.0
    # One side constant (zero variance) -> Pearson undefined -> 0.0.
    rng = np.random.default_rng(99)
    varying = rng.standard_normal((16, 16))
    assert V.phase_correlation(phase, varying) == 0.0
    assert V.phase_correlation(varying, phase) == 0.0


# --------------------------------------------------------------------------
# 2. Known phase: RMS == sigma, Strehl == exp(-sigma^2)
# --------------------------------------------------------------------------

def test_known_rms_and_marechal_strehl():
    """A phase map with known RMS sigma (piston removed) -> rms_wfe == sigma and
    strehl_marechal == exp(-sigma^2) (research/07 C.1)."""
    rng = np.random.default_rng(42)
    sigma = 0.3  # radians
    raw = rng.standard_normal((64, 64))
    # Normalise to exact population std = sigma after piston removal.
    raw = raw - raw.mean()
    raw = raw / raw.std()
    phase = sigma * raw + 5.0          # add piston: must be removed by the metric

    got_rms = V.rms_wfe(phase)
    assert got_rms == pytest.approx(sigma, rel=1e-12, abs=1e-12)

    # Strehl from the map and from the variance must agree, == exp(-sigma^2).
    expected_S = np.exp(-sigma ** 2)
    assert V.strehl_marechal(phase) == pytest.approx(expected_S, rel=1e-9)
    assert V.strehl_marechal(sigma ** 2) == pytest.approx(expected_S, rel=1e-12)


def test_strehl_scalar_variance_input():
    """strehl_marechal accepts a scalar residual variance and returns exp(-var)."""
    for var in (0.0, 0.01, 0.25, 1.0):
        assert V.strehl_marechal(var) == pytest.approx(np.exp(-var), rel=1e-12)
    assert V.strehl_marechal(0.0) == 1.0
    with pytest.raises(ValueError):
        V.strehl_marechal(-0.5)            # negative variance is invalid


def test_rad_to_meters_roundtrip():
    """rad_to_meters / meters_to_rad are inverse; lambda/2pi scaling (research/07
    C.1)."""
    lam = 500e-9
    sigma_rad = 0.4
    opd = V.rad_to_meters(sigma_rad, lam)
    assert opd == pytest.approx(lam / (2 * np.pi) * sigma_rad, rel=1e-12)
    assert V.meters_to_rad(opd, lam) == pytest.approx(sigma_rad, rel=1e-12)


# --------------------------------------------------------------------------
# 3. Phase correlation: +1 / -1 / ~0
# --------------------------------------------------------------------------

def test_phase_correlation_identical_is_one():
    """Identical maps -> rho == 1 (research/07 C.1)."""
    rng = np.random.default_rng(1)
    phase = rng.standard_normal((40, 40))
    assert V.phase_correlation(phase, phase) == pytest.approx(1.0, abs=1e-12)
    # Invariant to positive scale + piston.
    assert V.phase_correlation(2.0 * phase + 7.0, phase) == pytest.approx(
        1.0, abs=1e-12)


def test_phase_correlation_negative_is_minus_one():
    """A map vs its negative -> rho == -1 (research/07 C.1)."""
    rng = np.random.default_rng(2)
    phase = rng.standard_normal((40, 40))
    assert V.phase_correlation(phase, -phase) == pytest.approx(-1.0, abs=1e-12)


def test_phase_correlation_uncorrelated_near_zero():
    """Two independent random maps -> rho ~ 0 (statistical; research/07 C.1)."""
    rng = np.random.default_rng(3)
    a = rng.standard_normal((128, 128))
    b = rng.standard_normal((128, 128))
    rho = V.phase_correlation(a, b)
    # ~16k independent samples: |rho| ~ 1/sqrt(N) ~ 0.008; allow generous bound.
    assert abs(rho) < 0.05


def test_phase_correlation_with_mask():
    """Mask restricts the correlation to valid pupil pixels and stays in range."""
    rng = np.random.default_rng(4)
    a = rng.standard_normal((20, 20))
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    assert V.phase_correlation(a, a, mask=mask) == pytest.approx(1.0, abs=1e-12)
    assert V.phase_correlation(a, -a, mask=mask) == pytest.approx(-1.0, abs=1e-12)


# --------------------------------------------------------------------------
# 4. Residual variance
# --------------------------------------------------------------------------

def test_residual_variance_identical_is_zero():
    """residual_variance(phase, phase) == 0 (research/07 C.1)."""
    rng = np.random.default_rng(5)
    phase = rng.standard_normal((50, 50))
    assert V.residual_variance(phase, phase) == pytest.approx(0.0, abs=1e-20)


def test_residual_variance_vs_zero_is_var():
    """residual_variance(phase, 0) == population variance of phase (piston
    removed) (research/07 C.1)."""
    rng = np.random.default_rng(6)
    phase = rng.standard_normal((50, 50)) * 1.3 + 2.0
    zero = np.zeros_like(phase)
    expected = float(phase.var())          # numpy population variance (ddof=0)
    assert V.residual_variance(phase, zero) == pytest.approx(expected, rel=1e-12)


# --------------------------------------------------------------------------
# 5. r0 / tau0 recovery error
# --------------------------------------------------------------------------

def test_r0_recovery_error_percent():
    """r0_recovery_error(1.05, 1.0) == 5 % (relative); signed keeps the sign
    (research/07 C.1)."""
    assert V.r0_recovery_error(1.05, 1.0) == pytest.approx(5.0, rel=1e-12)
    assert V.r0_recovery_error(0.95, 1.0) == pytest.approx(5.0, rel=1e-12)
    assert V.r0_recovery_error(1.05, 1.0, signed=True) == pytest.approx(
        5.0, rel=1e-12)
    assert V.r0_recovery_error(0.95, 1.0, signed=True) == pytest.approx(
        -5.0, rel=1e-12)
    # Fraction (not percent).
    assert V.r0_recovery_error(1.05, 1.0, percent=False) == pytest.approx(
        0.05, rel=1e-12)
    with pytest.raises(ValueError):
        V.r0_recovery_error(0.1, 0.0)


def test_tau0_recovery_error_percent():
    """tau0_recovery_error mirrors r0 convention (research/07 C.1)."""
    assert V.tau0_recovery_error(0.0046, 0.0044) == pytest.approx(
        100.0 * (0.0046 - 0.0044) / 0.0044, rel=1e-9)
    assert V.tau0_recovery_error(0.004, 0.005, signed=True) == pytest.approx(
        -20.0, rel=1e-12)


# --------------------------------------------------------------------------
# 6. DM-corrected residual (the -1/2 reflection convention)
# --------------------------------------------------------------------------

def test_residual_after_dm_perfect_correction_identity():
    """commands == -true/2 with H = I -> wavefront correction -2*(H@cmd) == true
    -> residual ~ 0 (reflective DM, -1/2 convention; research/07 C.1, research/05
    S9)."""
    rng = np.random.default_rng(7)
    n = 50
    true_phase = rng.standard_normal(n)
    H = np.eye(n)
    commands = -true_phase / 2.0
    W_res, rms = V.residual_after_dm(true_phase, H, commands)
    assert rms == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(W_res, 0.0, atol=1e-12)


def test_residual_after_dm_sign_convention():
    """Explicit sign: residual == true + 2*(H@commands) (default reflection
    factor -0.5)."""
    rng = np.random.default_rng(8)
    n = 12
    true_phase = rng.standard_normal(n)
    H = rng.standard_normal((n, n))
    commands = rng.standard_normal(n)
    W_res, _ = V.residual_after_dm(true_phase, H, commands)
    expected = true_phase + 2.0 * (H @ commands)   # = true - (-2 * H@cmd)
    assert np.allclose(W_res, expected, atol=1e-12)


def test_dm_residual_realised_surface():
    """dm_residual differences an already-realised correction map and returns
    (rms, Strehl). Perfect correction -> rms 0, Strehl 1 (research/07 C.1)."""
    rng = np.random.default_rng(9)
    W_true = rng.standard_normal((16, 16)) * 0.2
    rms, strehl = V.dm_residual(W_true, W_true)          # perfect
    assert rms == pytest.approx(0.0, abs=1e-12)
    assert strehl == pytest.approx(1.0, abs=1e-12)
    # No correction -> residual == true; Strehl == exp(-var(true)).
    rms0, strehl0 = V.dm_residual(W_true, np.zeros_like(W_true))
    assert rms0 == pytest.approx(np.sqrt(W_true.var()), rel=1e-12)
    assert strehl0 == pytest.approx(np.exp(-W_true.var()), rel=1e-9)


# --------------------------------------------------------------------------
# 7. Zernike-coefficient decomposition error
# --------------------------------------------------------------------------

def test_zernike_decomp_error():
    """Per-mode and total coefficient error; orthonormal-basis L2 == wavefront
    RMS (research/07 C.3)."""
    true_c = np.array([0.0, 0.5, -0.3, 0.2, 0.1])
    est_c = np.array([0.0, 0.5, -0.3, 0.2, 0.1])
    err = V.zernike_decomp_error(true_c, est_c)
    assert err["total_rms"] == 0.0
    assert err["total_l2"] == 0.0
    assert np.allclose(err["per_mode"], 0.0)

    est_c2 = true_c + np.array([0.0, 0.0, 0.0, 0.0, 0.1])  # one mode off by 0.1
    err2 = V.zernike_decomp_error(true_c, est_c2)
    assert err2["max_abs"] == pytest.approx(0.1, rel=1e-12)
    assert err2["total_l2"] == pytest.approx(0.1, rel=1e-12)
    assert err2["per_mode"][4] == pytest.approx(0.1, rel=1e-12)
    with pytest.raises(ValueError):
        V.zernike_decomp_error(true_c, est_c2[:-1])         # length mismatch


# --------------------------------------------------------------------------
# 8. Error-budget summary
# --------------------------------------------------------------------------

def test_error_budget_keys_and_finite():
    """error_budget returns the expected keys, all finite (research/07 C.2)."""
    rng = np.random.default_rng(10)
    true = rng.standard_normal((48, 48)) * 0.25
    recon = true + rng.standard_normal((48, 48)) * 0.02
    eb = V.error_budget(recon, true, wavelength_m=500e-9,
                        r0_est=0.149, r0_true=0.15,
                        tau0_est=0.0046, tau0_true=0.0044)
    for key in ("rms_wfe_rad", "residual_variance_rad2", "strehl_marechal",
                "phase_correlation", "n_pix", "rms_wfe_m",
                "r0_recovery_error_pct", "tau0_recovery_error_pct"):
        assert key in eb, f"missing key {key}"
    for key, val in eb.items():
        assert np.isfinite(val), f"{key} not finite: {val}"
    # Sanity: low-noise reconstruction -> high Strehl, high correlation.
    assert eb["strehl_marechal"] > 0.9
    assert eb["phase_correlation"] > 0.95
    assert eb["n_pix"] == 48 * 48
    # Strehl must equal exp(-residual_variance).
    assert eb["strehl_marechal"] == pytest.approx(
        np.exp(-eb["residual_variance_rad2"]), rel=1e-9)
    # Recovery errors match the standalone helpers.
    assert eb["r0_recovery_error_pct"] == pytest.approx(
        V.r0_recovery_error(0.149, 0.15), rel=1e-12)


def test_error_budget_perfect_reconstruction():
    """Perfect reconstruction -> RMS 0, Strehl 1, correlation 1 (with a pupil
    mask)."""
    rng = np.random.default_rng(11)
    true = rng.standard_normal((32, 32))
    mask = np.zeros((32, 32), dtype=bool)
    yy, xx = np.mgrid[0:32, 0:32]
    mask[(xx - 15.5) ** 2 + (yy - 15.5) ** 2 <= 15.0 ** 2] = True
    eb = V.error_budget(true.copy(), true, mask=mask)
    assert eb["rms_wfe_rad"] == pytest.approx(0.0, abs=1e-12)
    assert eb["strehl_marechal"] == pytest.approx(1.0, abs=1e-12)
    # Correlation of a map with itself == 1.
    assert eb["phase_correlation"] == pytest.approx(1.0, abs=1e-12)


def test_error_budget_time_series_mean():
    """error_budget handles a stacked time-series (leading frame axis) and a
    per-frame mask broadcast across frames (research/07 C.2)."""
    rng = np.random.default_rng(12)
    T = 5
    true = rng.standard_normal((T, 24, 24)) * 0.3
    recon = true + rng.standard_normal((T, 24, 24)) * 0.03
    frame_mask = np.ones((24, 24), dtype=bool)
    eb = V.error_budget(recon, true, mask=frame_mask)
    assert eb["n_pix"] == T * 24 * 24
    assert np.isfinite(eb["rms_wfe_rad"])
    assert 0.0 < eb["strehl_marechal"] <= 1.0


# --------------------------------------------------------------------------
# 9. Reconstructor self-consistency & C/Python parity
# --------------------------------------------------------------------------

def test_reconstructor_self_consistency():
    """Gamma @ phi == slopes -> round-trip residual ~ 0 (research/07 C.2)."""
    rng = np.random.default_rng(13)
    Gamma = rng.standard_normal((30, 18))
    phi = rng.standard_normal(18)
    slopes = Gamma @ phi
    assert V.reconstructor_self_consistency(slopes, phi, Gamma) == pytest.approx(
        0.0, abs=1e-9)
    # A perturbed phase gives a positive residual.
    assert V.reconstructor_self_consistency(slopes, phi + 1.0, Gamma) > 0.0


def test_cpython_parity():
    """cpython_parity == max abs slope difference (research/07 C.2)."""
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([1.0, 2.0, 3.0, 4.0])
    assert V.cpython_parity(a, b) == 0.0
    c = b.copy()
    c[2] += 1e-6
    assert V.cpython_parity(a, c) == pytest.approx(1e-6, rel=1e-9)
    with pytest.raises(ValueError):
        V.cpython_parity(a, b[:-1])


def test_noll_variance_check_ratio_near_one():
    """Ensemble Zernike variances drawn at the Noll scale -> per-mode ratio ~1;
    piston (j=1) -> NaN (research/07 C.4)."""
    from aokit.zernike import noll_variance

    rng = np.random.default_rng(14)
    D_m, r0 = 1.0, 0.15
    scale = (D_m / r0) ** (5.0 / 3.0)
    n_modes = 8
    n_frames = 40000
    # Draw each mode k (j=k+1) as Gaussian with the theoretical variance.
    cols = []
    for k in range(n_modes):
        j = k + 1
        var = noll_variance(j) * scale
        std = np.sqrt(var)
        cols.append(rng.standard_normal(n_frames) * std)
    coeffs = np.column_stack(cols)
    ratio = V.noll_variance_check(coeffs, D_m, r0)
    assert ratio.shape == (n_modes,)
    assert np.isnan(ratio[0])                      # piston carries no power
    # Modes 2.. should recover ratio ~1 within Monte-Carlo scatter.
    finite = ratio[1:]
    assert np.all(np.isfinite(finite))
    assert np.allclose(finite, 1.0, atol=0.1)
