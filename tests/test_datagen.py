"""Unit tests for aokit.datagen (synthetic SH-WFS data generation backbone).

These tests verify the *validation backbone*: the generator injects a KNOWN
r0 / tau0 / Zernike content, and we confirm those values are recoverable from the
generated phase screens, spot fields, and frame series. research/07 PART B/C.

Coverage
--------
* FT / FT+subharmonics screens: structure-function r0 recovery within ~15-20%
  over an ensemble of seeds; log-log structure-function slope ~ 5/3.
* Zernike screen: returned coefficients carry the injected (D/r0)^(5/3) modal
  variance scaling -> recovered r0 ~ injected; deterministic per seed.
* Geometric spot field: a pure tip/tilt shifts ALL spots by the analytically
  predicted displacement (tilt * f / pixel_size).
* Fraunhofer spot field: produces light and the spot tracks an injected tilt.
* Frozen-flow series: frame k equals the base screen translated by wind*dt*k
  (within interpolation tolerance); injected tau0 = 0.314 r0 / v.
* add_noise: mean tracks the signal, variance follows shot-noise scaling, and
  the output dtype/bit-depth is correct.
* write_dataset: produces n_frames readable BMP frames and a valid
  ground_truth.json carrying the injected r0/tau0.
"""
import json
import os

import numpy as np
import pytest

from aokit.config import load_config
from aokit import datagen as dg
from aokit import geometry as geom
from aokit import zernike as zern
from aokit import bmpio


CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "example_config.json",
)


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def grid(cfg):
    return geom.build_subaperture_grid(cfg, valid_only=True)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _radial_structure_function(screen, delta_m, rmax_px=40):
    """Empirical phase structure function D_phi(r) = <(phi(x+r)-phi(x))^2>
    sampled along the x and y axes for integer lags 1..rmax_px-1.

    Returns ``(sep_m, D)`` (both 1-D, separations in metres).
    """
    seps = []
    Ds = []
    for lag in range(1, rmax_px):
        dx = screen[:, lag:] - screen[:, :-lag]
        dy = screen[lag:, :] - screen[:-lag, :]
        Ds.append(float(np.mean(dx ** 2)))
        seps.append(lag * delta_m)
        Ds.append(float(np.mean(dy ** 2)))
        seps.append(lag * delta_m)
    order = np.argsort(seps)
    return np.array(seps)[order], np.array(Ds)[order]


def _clipped_cog(frame, x0, y0, w):
    """Plain center-of-gravity over a (clipped) ``w x w`` window at top-left
    ``(x0, y0)``.  Returns ``(cx, cy)`` or ``(None, None)`` if no flux."""
    xi0 = max(int(np.floor(x0)), 0)
    yi0 = max(int(np.floor(y0)), 0)
    xi1 = min(xi0 + w, frame.shape[1])
    yi1 = min(yi0 + w, frame.shape[0])
    sub = frame[yi0:yi1, xi0:xi1]
    tot = float(sub.sum())
    if tot <= 0.0:
        return None, None
    xs = np.arange(xi0, xi1)
    ys = np.arange(yi0, yi1)
    cx = float((sub.sum(axis=0) * xs).sum() / tot)
    cy = float((sub.sum(axis=1) * ys).sum() / tot)
    return cx, cy


# ==========================================================================
# 1. FT / FT+subharmonics phase-screen statistics (r0 recovery)
# ==========================================================================

def test_ft_sh_screen_recovers_r0_via_structure_function():
    """FT+subharmonics: ensemble structure function D_phi(r)=6.88(r/r0)^5/3
    recovers the injected r0 within ~20% (statistical)."""
    r0_true = 0.15
    delta = 0.01          # m / px
    N = 256
    n_seeds = 16

    allD = []
    for s in range(n_seeds):
        scr = dg.ft_sh_phase_screen(r0_true, N, delta, L0_m=np.inf,
                                    n_subharmonics=5, seed=s)
        sep, D = _radial_structure_function(scr, delta)
        allD.append(D)
    Dmean = np.mean(allD, axis=0)

    # Invert D = 6.88 (r/r0)^(5/3) -> r0 = r * (6.88 / D)^(3/5) in the inertial
    # range (small r, away from the largest lags where the finite grid biases).
    m = (sep > 0.02) & (sep < 0.12)
    r0_est = float(np.median(sep[m] * (6.88 / Dmean[m]) ** (3.0 / 5.0)))

    print(f"\n[datagen] FT+sh r0_true={r0_true} r0_est={r0_est:.4f} "
          f"ratio={r0_est / r0_true:.3f}")
    assert 0.80 * r0_true <= r0_est <= 1.20 * r0_true


def test_ft_sh_structure_function_loglog_slope_is_5_3():
    """The log-log slope of the ensemble structure function is ~5/3
    (validates the Kolmogorov power law)."""
    r0_true = 0.15
    delta = 0.01
    N = 256

    allD = []
    for s in range(16):
        scr = dg.ft_sh_phase_screen(r0_true, N, delta, n_subharmonics=5, seed=s)
        sep, D = _radial_structure_function(scr, delta)
        allD.append(D)
    Dmean = np.mean(allD, axis=0)

    m = (sep > 0.02) & (sep < 0.20)
    slope = float(np.polyfit(np.log(sep[m]), np.log(Dmean[m]), 1)[0])
    print(f"[datagen] structure-function log-log slope = {slope:.3f} "
          f"(expect ~1.667)")
    assert 1.45 <= slope <= 1.85


def test_ft_screen_shape_and_finiteness():
    """Plain FFT screen has the requested shape, is finite and (essentially)
    zero-mean (piston unobservable)."""
    scr = dg.ft_phase_screen(0.12, 128, 0.01, seed=3)
    assert scr.shape == (128, 128)
    assert np.all(np.isfinite(scr))
    # Real-valued screen.
    assert scr.dtype == np.float64
    # Piston is dropped (DC PSD = 0): mean is small relative to the RMS.
    assert abs(scr.mean()) < 0.1 * scr.std()


def test_ft_screen_variance_scales_with_r0():
    """Smaller r0 (stronger turbulence) gives a larger phase variance."""
    delta, N = 0.01, 128
    var_strong = np.mean([dg.ft_sh_phase_screen(0.08, N, delta, seed=s).var()
                          for s in range(8)])
    var_weak = np.mean([dg.ft_sh_phase_screen(0.20, N, delta, seed=s).var()
                        for s in range(8)])
    print(f"[datagen] var(r0=0.08)={var_strong:.2f} > "
          f"var(r0=0.20)={var_weak:.2f}")
    assert var_strong > var_weak


# ==========================================================================
# 2. Zernike-synthesis screen (known modal content -> r0 recovery)
# ==========================================================================

def test_zernike_screen_deterministic_per_seed():
    """Same seed -> identical coefficients and screen; different seed differs."""
    s1, c1 = dg.zernike_phase_screen(0.2, 1.0, 48, 15, seed=7)
    s2, c2 = dg.zernike_phase_screen(0.2, 1.0, 48, 15, seed=7)
    s3, c3 = dg.zernike_phase_screen(0.2, 1.0, 48, 15, seed=8)
    assert np.allclose(c1, c2) and np.allclose(s1, s2)
    assert not np.allclose(c1, c3)
    # Coefficient vector length is j_max - 1 (Noll j = 2..15).
    assert c1.shape == (14,)


def test_zernike_screen_coeff_variance_recovers_r0():
    """Ensemble Zernike-coefficient variances follow <a_j^2>=c_j (D/r0)^(5/3);
    inverting the mid-order modes recovers the injected r0 to a few %."""
    D = 1.0
    r0_true = 0.2
    j_max = 15
    n_ens = 400

    js = list(range(2, j_max + 1))
    acc = np.zeros(len(js))
    for s in range(n_ens):
        _, c = dg.zernike_phase_screen(r0_true, D, 48, j_max, seed=s)
        acc += c ** 2
    var = acc / n_ens

    cj = np.array([zern.noll_variance(j) for j in js])
    # Use defocus-and-up (j >= 4): tip/tilt dominate the total but the law holds
    # per mode; (D/r0)^(5/3) = var_j / c_j.
    mid = np.array([j >= 4 for j in js])
    scale_est = float(np.median(var[mid] / cj[mid]))   # estimates (D/r0)^(5/3)
    r0_est = D * (1.0 / scale_est) ** (3.0 / 5.0)

    print(f"\n[datagen] Zernike r0_true={r0_true} r0_est={r0_est:.4f} "
          f"ratio={r0_est / r0_true:.3f}")
    assert 0.85 * r0_true <= r0_est <= 1.15 * r0_true


def test_zernike_screen_low_order_variance_scaling():
    """Low-order coefficient variances scale ~ (D/r0)^(5/3) between two r0:
    the variance ratio matches the analytic ratio."""
    D, j_max, n_ens = 1.0, 11, 300
    js = list(range(2, j_max + 1))

    def ens_var(r0):
        acc = np.zeros(len(js))
        for s in range(n_ens):
            _, c = dg.zernike_phase_screen(r0, D, 40, j_max, seed=s)
            acc += c ** 2
        return acc / n_ens

    v_a = ens_var(0.10)
    v_b = ens_var(0.20)
    # Theoretical variance ratio = (r0_b / r0_a)^(5/3) = (0.2/0.1)^(5/3).
    theo = (0.20 / 0.10) ** (5.0 / 3.0)
    meas = float(np.median(v_a / v_b))    # v(small r0) / v(large r0) > 1
    print(f"[datagen] var-ratio measured={meas:.3f} theory={theo:.3f}")
    assert 0.75 * theo <= meas <= 1.25 * theo


# ==========================================================================
# 3. Geometric spot field (analytic tip/tilt oracle)
# ==========================================================================

def test_spots_geometric_tiptilt_shifts_all_spots(cfg, grid):
    """A pure x-tilt phase ramp shifts every (interior) spot by the analytic
    displacement delta = f * theta / pixel_size, theta = (lambda/2pi)<dphi/dx>."""
    H, W = cfg.camera.frame_h, cfg.camera.frame_w
    a = 0.3                       # rad / pixel ramp -> measurable shift
    xx = np.arange(W)[None, :].repeat(H, 0).astype(float)
    phase = a * xx

    img = dg.spots_geometric(cfg, phase, grid=grid, spot_sigma_px=1.5)

    lam = cfg.wavelength_m
    f = cfg.mla.focal_length_m
    px = cfg.camera.pixel_size_m
    # <dphi/dx_metres> = a / px;  theta = (lambda/2pi)*(a/px);
    # delta_px = f*theta/px = f*(lambda/2pi)*a/px^2.
    expected_dx = f * (lam / (2.0 * np.pi)) * a / (px * px)
    assert expected_dx > 2.0      # sanity: the test tilt is clearly measurable

    errs = []
    for k in range(grid.ref_x.shape[0]):
        cx, cy = _clipped_cog(img, grid.ref_x[k] - grid.w / 2.0,
                              grid.ref_y[k] - grid.w / 2.0, grid.w)
        if cx is None:
            continue
        dx = cx - grid.ref_x[k]
        dy = cy - grid.ref_y[k]
        # y-shift should be ~0 for a pure x-tilt.
        assert abs(dy) < 0.5
        errs.append(abs(dx - expected_dx))
    errs = np.array(errs)
    print(f"\n[datagen] geometric tip/tilt expected dx={expected_dx:.3f} px, "
          f"mean|err|={errs.mean():.4f} median|err|={np.median(errs):.4f}")
    # The bulk of sub-apertures match the analytic shift to well under a pixel
    # (edge spots that clip the frame are excluded by the median criterion).
    assert np.median(errs) < 0.25


def test_spots_geometric_specific_subapertures(cfg, grid):
    """Spot-check two individual interior sub-apertures against the analytic
    tip displacement (not just the ensemble)."""
    H, W = cfg.camera.frame_h, cfg.camera.frame_w
    a = 0.2
    yy = np.arange(H)[:, None].repeat(W, 1).astype(float)
    phase = a * yy                        # pure y-tilt this time

    img = dg.spots_geometric(cfg, phase, grid=grid, spot_sigma_px=1.5)
    lam, f, px = cfg.wavelength_m, cfg.mla.focal_length_m, cfg.camera.pixel_size_m
    expected_dy = f * (lam / (2.0 * np.pi)) * a / (px * px)

    # Pick two sub-apertures near the pupil centre (well inside the frame).
    cx0, cy0 = cfg.pupil.center_x_px, cfg.pupil.center_y_px
    d2 = (grid.ref_x - cx0) ** 2 + (grid.ref_y - cy0) ** 2
    central = np.argsort(d2)[:2]
    for k in central:
        cx, cy = _clipped_cog(img, grid.ref_x[k] - grid.w / 2.0,
                              grid.ref_y[k] - grid.w / 2.0, grid.w)
        assert cx is not None
        assert abs((cy - grid.ref_y[k]) - expected_dy) < 0.25
        assert abs(cx - grid.ref_x[k]) < 0.5     # no x-shift for a y-tilt


def test_spots_geometric_flat_phase_at_reference(cfg, grid):
    """A flat (zero) phase puts every spot at its reference position."""
    H, W = cfg.camera.frame_h, cfg.camera.frame_w
    img = dg.spots_geometric(cfg, np.zeros((H, W)), grid=grid, spot_sigma_px=1.5)
    cx0, cy0 = cfg.pupil.center_x_px, cfg.pupil.center_y_px
    d2 = (grid.ref_x - cx0) ** 2 + (grid.ref_y - cy0) ** 2
    for k in np.argsort(d2)[:5]:
        cx, cy = _clipped_cog(img, grid.ref_x[k] - grid.w / 2.0,
                              grid.ref_y[k] - grid.w / 2.0, grid.w)
        assert abs(cx - grid.ref_x[k]) < 0.1
        assert abs(cy - grid.ref_y[k]) < 0.1


# ==========================================================================
# 4. Fraunhofer spot field
# ==========================================================================

def test_spots_fraunhofer_produces_light_and_tracks_tilt(cfg, grid):
    """The Fraunhofer model yields a non-empty frame and the spot moves in the
    tilt direction (physically faithful, looser tolerance than geometric)."""
    H, W = cfg.camera.frame_h, cfg.camera.frame_w
    img_flat = dg.spots_fraunhofer(cfg, np.zeros((H, W)), grid=grid)
    assert img_flat.shape == (H, W)
    assert img_flat.sum() > 0.0

    a = 0.3
    xx = np.arange(W)[None, :].repeat(H, 0).astype(float)
    img_tilt = dg.spots_fraunhofer(cfg, a * xx, grid=grid)

    cx0, cy0 = cfg.pupil.center_x_px, cfg.pupil.center_y_px
    k = int(np.argmin((grid.ref_x - cx0) ** 2 + (grid.ref_y - cy0) ** 2))
    c_flat = _clipped_cog(img_flat, grid.ref_x[k] - grid.w / 2.0,
                          grid.ref_y[k] - grid.w / 2.0, grid.w)
    c_tilt = _clipped_cog(img_tilt, grid.ref_x[k] - grid.w / 2.0,
                          grid.ref_y[k] - grid.w / 2.0, grid.w)
    # Spot shifts toward +x with a positive x-tilt.
    assert c_tilt[0] - c_flat[0] > 1.0


# ==========================================================================
# 5. Frozen-flow series (known tau0 = 0.314 r0 / v)
# ==========================================================================

def test_frozen_flow_frames_are_translated_base(cfg):
    """Frame k equals the base screen translated by wind*dt*k (within bilinear
    interpolation tolerance)."""
    r0, wind = 0.15, 2.0
    series, gt = dg.frozen_flow_series(cfg, r0, wind, 0.0, 5,
                                       n_grid=128, seed=1)
    sx, sy = gt["shift_xy_px_per_frame"]
    base = series[0]
    for k in range(5):
        manual = dg._shift_screen(base, sx * k, sy * k)
        # Exact: the generator shifts the SAME base screen by k*shift.
        assert np.max(np.abs(series[k] - manual)) < 1e-9


def test_frozen_flow_tau0_matches_injected(cfg):
    """The reported tau0 equals the Taylor frozen-flow value 0.314 r0 / v."""
    r0, wind = 0.18, 7.5
    _, gt = dg.frozen_flow_series(cfg, r0, wind, 30.0, 3, n_grid=64, seed=0)
    expected_tau0 = 0.314 * r0 / wind
    assert gt["tau0_s"] == pytest.approx(expected_tau0, rel=1e-9)
    # Per-frame pixel shift = v*dt/pixel_size along the wind vector.
    shift = wind * cfg.dt_s / cfg.camera.pixel_size_m
    assert gt["shift_px_per_frame"] == pytest.approx(shift, rel=1e-9)


def test_frozen_flow_subpixel_shift_interpolates(cfg):
    """A known sub-pixel shift recovers a translated version of a smooth ramp
    (interpolation correctness)."""
    # Smooth low-frequency screen so bilinear interpolation is accurate.
    n = 64
    lin = np.linspace(0, 2 * np.pi, n)
    xx, _ = np.meshgrid(lin, lin)
    base = np.sin(xx)                          # smooth in x
    shifted = dg._shift_screen(base, 0.5, 0.0)  # half-pixel in x
    # Interior comparison vs analytic sin shifted by half a pixel.
    dx = (lin[1] - lin[0]) * 0.5
    analytic = np.sin(xx - dx)
    assert np.max(np.abs(shifted[:, 2:-2] - analytic[:, 2:-2])) < 0.02


# ==========================================================================
# 6. Detector noise model
# ==========================================================================

def test_add_noise_dtype_and_range():
    """8-bit output is uint8 in [0,255]; 16-bit output is uint16."""
    img = np.zeros((32, 32))
    img[16, 16] = 1.0
    out8 = dg.apply_detector_noise(img, flux_photons=1e4, bit_depth=8, seed=0)
    assert out8.dtype == np.uint8
    assert out8.min() >= 0 and out8.max() <= 255
    out16 = dg.apply_detector_noise(img, flux_photons=1e4, bit_depth=16, seed=0)
    assert out16.dtype == np.uint16


def test_add_noise_mean_tracks_signal():
    """With no read noise / gain=1, the mean electrons at a pixel equal the
    expected photo-electrons (Poisson is unbiased)."""
    img = np.zeros((16, 16))
    img[8, 8] = 1.0                       # all flux in one pixel
    flux = 5000.0
    qe = 0.9
    vals = [dg.apply_detector_noise(img, flux_photons=flux, qe=qe,
                                    read_noise_e=0.0, gain=1.0, bias=0.0,
                                    bit_depth=16, seed=s)[8, 8]
            for s in range(300)]
    mean = float(np.mean(vals))
    expected = flux * qe                  # img normalised to total 1
    assert mean == pytest.approx(expected, rel=0.05)


def test_add_noise_shot_noise_scaling():
    """Shot noise: variance ~ mean (Poisson), and variance grows with the photon
    count (lower flux -> lower absolute variance, ~equal relative variance)."""
    img = np.zeros((16, 16))
    img[8, 8] = 1.0

    def mean_var(flux):
        vals = np.array([dg.apply_detector_noise(
            img, flux_photons=flux, qe=1.0, read_noise_e=0.0, gain=1.0,
            bias=0.0, bit_depth=16, seed=s)[8, 8] for s in range(400)],
            dtype=float)
        return vals.mean(), vals.var()

    m_lo, v_lo = mean_var(1.0e3)
    m_hi, v_hi = mean_var(1.0e4)
    # Poisson: variance approximately equals the mean.
    assert v_lo == pytest.approx(m_lo, rel=0.2)
    assert v_hi == pytest.approx(m_hi, rel=0.2)
    # Absolute variance increases with photon count.
    assert v_hi > v_lo
    print(f"\n[datagen] shot noise: flux1e3 var/mean={v_lo / m_lo:.3f}, "
          f"flux1e4 var/mean={v_hi / m_hi:.3f}")


def test_add_noise_read_noise_adds_floor():
    """Read noise raises the variance of a zero-signal pixel above the (zero)
    shot-noise floor."""
    img = np.zeros((16, 16))            # no signal anywhere
    # Bias so the read-noise fluctuations are not clipped at 0.
    vals = np.array([dg.apply_detector_noise(
        img, flux_photons=0.0, read_noise_e=5.0, gain=1.0, bias=50.0,
        bit_depth=16, seed=s)[0, 0] for s in range(400)], dtype=float)
    assert vals.std() > 1.0            # read noise present
    assert vals.mean() == pytest.approx(50.0, abs=2.0)   # centred on bias


# ==========================================================================
# 7. End-to-end dataset writer (the validation oracle)
# ==========================================================================

def test_write_dataset_produces_readable_bmps_and_ground_truth(cfg, tmp_path):
    """generate_dataset writes n_frames readable BMP frames and a valid
    ground_truth.json carrying the injected r0/tau0 and the config."""
    r0 = 0.15
    tau0 = 0.045
    n_frames = 4
    out = str(tmp_path / "run01")

    gt = dg.generate_dataset(cfg, r0, tau0, n_frames, out,
                             spot_model="geometric", j_max=6, seed=2)

    # n_frames BMP files, zero-padded and chronologically sortable.
    bmps = sorted(f for f in os.listdir(out) if f.endswith(".bmp"))
    assert len(bmps) == n_frames
    assert bmps[0] == "frame_0000.bmp"

    # Each BMP is readable by the project's reader at the configured resolution.
    for name in bmps:
        frame = bmpio.read_bmp_gray(os.path.join(out, name))
        assert frame.shape == (cfg.camera.frame_h, cfg.camera.frame_w)
        assert np.all(np.isfinite(frame))

    # ground_truth.json exists, parses, and carries the injected truth.
    gt_path = os.path.join(out, "ground_truth.json")
    assert os.path.exists(gt_path)
    with open(gt_path) as fh:
        disk = json.load(fh)
    assert disk["r0_m"] == pytest.approx(r0)
    assert disk["tau0_s"] == pytest.approx(tau0)
    assert disk["wavelength_m"] == pytest.approx(cfg.wavelength_m)
    # Wind reconciled from tau0: v = 0.314 r0 / tau0.
    assert disk["wind_speed_mps"] == pytest.approx(0.314 * r0 / tau0, rel=1e-6)
    assert disk["n_frames"] == n_frames
    # Per-frame Zernike ground truth present (j_max>1).
    assert len(disk["zernike_noll_per_frame"]) == n_frames
    assert len(disk["zernike_noll_per_frame"][0]) == 6 - 1   # j = 2..6
    # Config round-trips the key fields.
    assert disk["config"]["mla"]["n_lenslets_x"] == cfg.mla.n_lenslets_x

    # The returned manifest matches what was written.
    assert gt["r0_m"] == disk["r0_m"]
    assert gt["frames"] == bmps


def test_write_dataset_fraunhofer_model(cfg, tmp_path):
    """The Fraunhofer spot model also produces a complete, readable dataset."""
    out = str(tmp_path / "run_fh")
    gt = dg.generate_dataset(cfg, 0.15, 0.045, 2, out,
                             spot_model="fraunhofer", seed=5)
    bmps = sorted(f for f in os.listdir(out) if f.endswith(".bmp"))
    assert len(bmps) == 2
    frame = bmpio.read_bmp_gray(os.path.join(out, bmps[0]))
    assert frame.shape == (cfg.camera.frame_h, cfg.camera.frame_w)
    assert frame.max() > 0
    assert gt["spot_model"] == "fraunhofer"
