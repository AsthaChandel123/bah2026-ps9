"""Unit tests for aokit.centroiding (CoG family + TWCoG + driver).

These tests build small synthetic sub-images directly (a known Gaussian spot at
a known sub-pixel center) so they do NOT depend on aokit.geometry being
implemented. Tolerances follow research/01: CoG is coarse, TWCoG / Gaussian-fit
are tight; thresholding wins under background; reference subtraction cancels
common-mode bias.

Coordinate convention (research/01 S1.1; matches aokit.centroiding):
    (cx, cy) = (column / x, row / y) of the spot in window-local pixels.
"""
import numpy as np
import pytest
from dataclasses import dataclass

import aokit.centroiding as C


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def render_gaussian(h, w, cx, cy, sigma=1.8, amp=1000.0, bg=0.0):
    """Render a 2-D Gaussian spot centered at (cx, cy)=(col, row).

    Returns an ``(h, w)`` float64 array ``I[row, col]``.
    """
    cols = np.arange(w, dtype=np.float64)[None, :]
    rows = np.arange(h, dtype=np.float64)[:, None]
    r2 = (cols - cx) ** 2 + (rows - cy) ** 2
    return bg + amp * np.exp(-r2 / (2.0 * sigma * sigma))


def render_elong(h, w, cx, cy, sx, sy, amp=1000.0, bg=0.0):
    """Render an elongated (anisotropic) Gaussian; widths (sx, sy) in (col, row)."""
    cols = np.arange(w, dtype=np.float64)[None, :]
    rows = np.arange(h, dtype=np.float64)[:, None]
    return bg + amp * np.exp(-((cols - cx) ** 2 / (2 * sx * sx)
                               + (rows - cy) ** 2 / (2 * sy * sy)))


def err(est, true):
    return abs(est[0] - true[0]), abs(est[1] - true[1])


# Documented tolerances (pixels), noiseless well-sampled spot.
TOL_COG = 0.05         # plain CoG, coarse
TOL_TCOG = 0.02
TOL_WCOG = 0.05
TOL_TWCOG = 0.01       # primary estimator, tight
TOL_IWCOG = 0.01
TOL_BRIGHT = 0.03
TOL_CORR = 0.02        # correlation on a compact spot
TOL_GAUSSFIT = 1e-3    # ground-truth oracle, very tight


# ---------------------------------------------------------------------------
# Per-method recovery on a known sub-pixel center
# ---------------------------------------------------------------------------

# A spread of known sub-pixel centers (col, row).
KNOWN_CENTERS = [(8.0, 8.0), (7.3, 8.65), (9.15, 6.4), (6.6, 9.35), (8.5, 8.5)]


@pytest.mark.parametrize("cx,cy", KNOWN_CENTERS)
def test_cog_recovers_known_subpixel_position(cx, cy):
    """A synthetic Gaussian spot at a known sub-pixel (cx,cy) is recovered by
    cog() to < 0.05 px in the noiseless, well-sampled case (research/01 M1)."""
    img = render_gaussian(18, 18, cx, cy, sigma=1.8, amp=1000.0)
    ex, ey = err(C.cog(img), (cx, cy))
    assert ex < TOL_COG and ey < TOL_COG, f"CoG err=({ex},{ey})"
    # The brief's alias must agree exactly with cog().
    assert C.center_of_gravity(img) == C.cog(img)


@pytest.mark.parametrize("cx,cy", KNOWN_CENTERS)
def test_thresholded_cog_recovers_known_position(cx, cy):
    """thresholded_cog() recovers the center on a clean spot (research/01 M2)."""
    img = render_gaussian(18, 18, cx, cy, sigma=1.8, amp=1000.0)
    ex, ey = err(C.thresholded_cog(img, frac=0.05), (cx, cy))
    assert ex < TOL_TCOG and ey < TOL_TCOG, f"TCoG err=({ex},{ey})"


@pytest.mark.parametrize("cx,cy", KNOWN_CENTERS)
def test_weighted_cog_recovers_known_position(cx, cy):
    """weighted_cog() with a Gaussian weight recovers the center; a precomputed
    weight LUT gives the same answer as the sigma form (research/01 M3)."""
    img = render_gaussian(18, 18, cx, cy, sigma=1.8, amp=1000.0)
    # Weight re-centered near the spot (use a precomputed LUT, the C-core form).
    cx0, cy0 = C.thresholded_cog(img, frac=0.1)
    wts = C.gaussian_weight(18, 18, cx0, cy0, fwhm_px=4.5)
    ex, ey = err(C.weighted_cog(img, weights=wts), (cx, cy))
    assert ex < TOL_WCOG and ey < TOL_WCOG, f"WCoG err=({ex},{ey})"


@pytest.mark.parametrize("cx,cy", KNOWN_CENTERS)
def test_twcog_recovers_known_position_tight(cx, cy):
    """TWCoG (the PRIMARY estimator) recovers the center tightly on a clean,
    well-sampled spot (research/01 S12)."""
    img = render_gaussian(18, 18, cx, cy, sigma=1.8, amp=1000.0)
    ex, ey = err(C.twcog(img, thresh_frac=0.05, fwhm_px=4.2), (cx, cy))
    assert ex < TOL_TWCOG and ey < TOL_TWCOG, f"TWCoG err=({ex},{ey})"


@pytest.mark.parametrize("cx,cy", KNOWN_CENTERS)
def test_iter_weighted_cog_recovers_known_position(cx, cy):
    """Iteratively-weighted CoG converges to the center (research/01 M4)."""
    img = render_gaussian(18, 18, cx, cy, sigma=1.8, amp=1000.0)
    ex, ey = err(C.iter_weighted_cog(img, sigma_w=2.0, n_iter=3), (cx, cy))
    assert ex < TOL_IWCOG and ey < TOL_IWCOG, f"IWCoG err=({ex},{ey})"


@pytest.mark.parametrize("cx,cy", KNOWN_CENTERS)
def test_brightest_pixel_recovers_known_position(cx, cy):
    """Brightest-pixel selection + CoG recovers the center (research/01 M7)."""
    img = render_gaussian(18, 18, cx, cy, sigma=1.8, amp=1000.0)
    ex, ey = err(C.brightest_pixel(img, n_bright=29), (cx, cy))
    assert ex < TOL_BRIGHT and ey < TOL_BRIGHT, f"brightest err=({ex},{ey})"


@pytest.mark.parametrize("cx,cy", KNOWN_CENTERS)
def test_correlation_centroid_compact_spot(cx, cy):
    """correlation_centroid() with a matched template recovers a compact-spot
    center; the parabolic sub-pixel step keeps the bias small (research/01 M10).
    """
    tmpl = render_gaussian(18, 18, 9.0, 9.0, sigma=1.8, amp=1000.0)
    img = render_gaussian(18, 18, cx, cy, sigma=1.8, amp=1000.0)
    ex, ey = err(C.correlation_centroid(img, tmpl), (cx, cy))
    assert ex < TOL_CORR and ey < TOL_CORR, f"correlation err=({ex},{ey})"


@pytest.mark.parametrize("cx,cy", KNOWN_CENTERS)
def test_gaussfit_is_ground_truth_reference(cx, cy):
    """gaussfit_centroid() matches the injected position to high precision on a
    true Gaussian spot (offline oracle, research/01 M9)."""
    img = render_gaussian(20, 20, cx, cy, sigma=2.0, amp=1000.0, bg=5.0)
    ex, ey = err(C.gaussfit_centroid(img), (cx, cy))
    assert ex < TOL_GAUSSFIT and ey < TOL_GAUSSFIT, f"gaussfit err=({ex},{ey})"


# ---------------------------------------------------------------------------
# Background-offset: thresholding beats plain CoG
# ---------------------------------------------------------------------------

def test_thresholded_cog_beats_cog_under_background():
    """With a DC background pedestal, thresholded_cog() recovers the true center
    far better than plain cog() (research/01 M2, S2.1)."""
    cx, cy = 8.4, 7.6
    spot = render_gaussian(18, 18, cx, cy, sigma=1.8, amp=800.0)
    bg = 120.0
    img = spot + bg

    e_cog = max(err(C.cog(img), (cx, cy)))
    e_tcog = max(err(C.thresholded_cog(img, thresh=bg * 1.2), (cx, cy)))

    assert e_tcog < 0.02, f"TCoG should be accurate under bg; got {e_tcog}"
    assert e_tcog < 0.1 * e_cog, (
        f"TCoG ({e_tcog}) should hugely beat CoG ({e_cog}) under DC background")


def test_thresholded_cog_frac_handles_background():
    """A fraction-of-max threshold also suppresses a moderate DC pedestal."""
    cx, cy = 7.7, 9.2
    img = render_gaussian(18, 18, cx, cy, sigma=1.8, amp=600.0) + 40.0
    # frac*max ~ 0.1*640 = 64 > pedestal 40, so the pedestal is removed.
    e = max(err(C.thresholded_cog(img, frac=0.12), (cx, cy)))
    assert e < 0.03, f"frac-threshold under background, err={e}"


# ---------------------------------------------------------------------------
# Shift test: recovered displacement ~= injected (dx, dy)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dx,dy", [(0.0, 0.0), (1.0, 0.0), (0.0, -1.0),
                                   (0.7, -0.4), (-0.6, 0.9), (1.3, 1.1)])
def test_shift_recovered_as_displacement(dx, dy):
    """A spot shifted by (dx,dy) from a reference yields recovered displacement
    ~= (dx,dy) for TWCoG (the slope is centroid - reference)."""
    base_x, base_y = 9.0, 9.0
    ref = render_gaussian(18, 18, base_x, base_y, sigma=1.8, amp=1000.0)
    shifted = render_gaussian(18, 18, base_x + dx, base_y + dy,
                              sigma=1.8, amp=1000.0)

    rcx, rcy = C.twcog(ref, thresh_frac=0.05, fwhm_px=4.2)
    scx, scy = C.twcog(shifted, thresh_frac=0.05, fwhm_px=4.2)
    rec_dx, rec_dy = scx - rcx, scy - rcy

    assert abs(rec_dx - dx) < 0.01, f"dx: rec={rec_dx} true={dx}"
    assert abs(rec_dy - dy) < 0.01, f"dy: rec={rec_dy} true={dy}"


# ---------------------------------------------------------------------------
# Reference subtraction cancels common-mode bias
# ---------------------------------------------------------------------------

def test_reference_subtraction_cancels_common_mode_bias():
    """Using the SAME (biased) estimator for the reference cancels the *constant*
    (common-mode) part of the weighting center-pull bias in
    (centroid - reference) (research/01 S5, S8).

    A fixed-center weighted CoG pulls an off-center spot toward the window
    center -- a large absolute bias. The reference spot sits at (nearly) the same
    offset, so that absolute bias is common-mode: (spot - reference) removes it,
    leaving only the small residual from the bias *gradient* across the tiny
    relative shift. The differential displacement error is therefore far smaller
    than the reference's own absolute bias that got cancelled."""
    base = (10.8, 6.2)       # well off the window center (8.5,8.5) -> biased WCoG
    dx, dy = 0.25, 0.30      # small relative shift (slope regime)
    spot = render_gaussian(18, 18, base[0] + dx, base[1] + dy,
                           sigma=1.8, amp=1000.0)
    ref = render_gaussian(18, 18, base[0], base[1], sigma=1.8, amp=1000.0)

    # Fixed-center Gaussian weight at the WINDOW center -> deliberately biased.
    wts = C.gaussian_weight(18, 18, 8.5, 8.5, fwhm_px=4.0)

    # Absolute (common-mode) bias carried by the reference centroid.
    rcx, rcy = C.weighted_cog(ref, weights=wts)
    ref_abs_bias = max(abs(rcx - base[0]), abs(rcy - base[1]))

    # Differential displacement after reference subtraction.
    scx, scy = C.weighted_cog(spot, weights=wts)
    rel_err = max(abs((scx - rcx) - dx), abs((scy - rcy) - dy))

    # The fixed-center weight carries a large absolute bias on each centroid...
    assert ref_abs_bias > 0.3, (
        f"expected a sizeable common-mode WCoG bias to cancel; got {ref_abs_bias}")
    # ...and reference subtraction removes most of it: the differential error is
    # a small fraction of the absolute (common-mode) bias that was cancelled.
    assert rel_err < 0.5 * ref_abs_bias, (
        f"reference subtraction should cancel the common-mode bias: "
        f"rel_err={rel_err}, ref_abs_bias={ref_abs_bias}")

    # Cross-check: TWCoG (which re-centers the weight on the spot) is itself
    # nearly unbiased here, recovering the shift tightly even without the trick.
    s2 = C.twcog(spot, thresh_frac=0.05, fwhm_px=4.2)
    r2 = C.twcog(ref, thresh_frac=0.05, fwhm_px=4.2)
    tw_rel_err = max(abs((s2[0] - r2[0]) - dx), abs((s2[1] - r2[1]) - dy))
    assert tw_rel_err < 0.02, f"TWCoG differential shift error {tw_rel_err}"


# ---------------------------------------------------------------------------
# Correlation beats CoG for extended/elongated (truncated) spots
# ---------------------------------------------------------------------------

def test_correlation_beats_cog_for_extended_truncated_spot():
    """For an elongated spot whose long tail is clipped by the window edge, CoG
    is biased by the truncation, while correlation_centroid() with a template of
    the full elongated shape locates the true center far more accurately
    (research/01 M10)."""
    H = W = 20
    sx, sy = 3.0, 1.8     # elongated along x
    true_cx, true_cy = 14.0, 10.0   # near the right edge -> right tail clipped

    tmpl = render_elong(H, W, 9.5, 9.5, sx, sy, amp=1000.0)
    img = render_elong(H, W, true_cx, true_cy, sx, sy, amp=1000.0)

    e_cog = abs(C.cog(img)[0] - true_cx)               # x is the truncated axis
    e_corr = abs(C.correlation_centroid(img, tmpl)[0] - true_cx)

    assert e_cog > 0.15, f"expected CoG to be biased by truncation; got {e_cog}"
    assert e_corr < e_cog, (
        f"correlation ({e_corr}) should beat CoG ({e_cog}) on extended spot")
    assert e_corr < 0.5 * e_cog, (
        f"correlation should beat CoG by a clear margin: "
        f"e_corr={e_corr}, e_cog={e_cog}")


# ---------------------------------------------------------------------------
# Noise robustness: bias stays small over many trials
# ---------------------------------------------------------------------------

def test_noise_robustness_bias_small_over_many_trials():
    """With Poisson shot + Gaussian read noise, the mean centroid error (bias)
    stays small over many trials for CoG, TWCoG and Gaussian-fit; TWCoG/CoG
    scatter is near the photon-noise floor (research/01 S2.1-S2.3)."""
    rng = np.random.default_rng(1234)
    n_trials = 200
    read = 8.0
    amp = 2500.0
    thr = 3.0 * read  # noise-floor threshold (m*sigma_read, m=3)

    methods = {
        "cog_thr": lambda im: C.thresholded_cog(im, thresh=thr),
        "twcog": lambda im: C.twcog(im, thresh_sigma=thr, thresh_frac=0.0,
                                    fwhm_px=4.2),
        "gaussfit": lambda im: C.gaussfit_centroid(im),
    }
    errs = {k: [] for k in methods}

    for _ in range(n_trials):
        cx = 9.0 + rng.uniform(-2.0, 2.0)
        cy = 9.0 + rng.uniform(-2.0, 2.0)
        clean = render_gaussian(18, 18, cx, cy, sigma=1.8, amp=amp)
        noisy = rng.poisson(np.clip(clean, 0, None)).astype(np.float64)
        noisy += rng.normal(0.0, read, noisy.shape)
        for k, fn in methods.items():
            ex, ey = fn(noisy)
            if np.isfinite(ex) and np.isfinite(ey):
                errs[k].append((ex - cx, ey - cy))

    for k, e in errs.items():
        e = np.asarray(e)
        assert e.shape[0] > 0.9 * n_trials, f"{k}: too many failures"
        bias = np.abs(e.mean(axis=0))
        scatter = e.std(axis=0)
        assert np.all(bias < 0.02), f"{k}: bias too large: {bias}"
        assert np.all(scatter < 0.1), f"{k}: scatter too large: {scatter}"


# ---------------------------------------------------------------------------
# Robustness: zero-flux / dead window returns NaN, never raises
# ---------------------------------------------------------------------------

def test_zero_flux_window_returns_nan():
    """A zero-flux (dead) window returns (nan, nan) for every estimator rather
    than raising or dividing by zero."""
    dead = np.zeros((16, 16), dtype=np.float64)
    for est in (C.cog,
                lambda im: C.thresholded_cog(im, frac=0.05),
                lambda im: C.weighted_cog(im, sigma=2.0),
                lambda im: C.twcog(im, thresh_frac=0.05),
                lambda im: C.iter_weighted_cog(im, sigma_w=2.0),
                lambda im: C.brightest_pixel(im, n_bright=9)):
        cx, cy = est(dead)
        assert np.isnan(cx) and np.isnan(cy)


def test_twcog_validity_gate_rejects_underfilled_window():
    """The TWCoG >=min_pixels validity gate returns NaN when too few pixels are
    above threshold (research/01 S4, S12)."""
    img = np.zeros((16, 16), dtype=np.float64)
    img[8, 8] = 1000.0   # a single hot pixel
    cx, cy = C.twcog(img, thresh_frac=0.5, min_pixels=3)
    assert np.isnan(cx) and np.isnan(cy)


# ---------------------------------------------------------------------------
# Driver: frame + sub-aperture grid -> centroids -> canonical slope vector
# ---------------------------------------------------------------------------

@dataclass
class _FakeGrid:
    """Minimal duck-typed stand-in for aokit.geometry.SubApertureGrid so the
    driver tests don't depend on geometry.py being implemented."""
    x0: np.ndarray
    y0: np.ndarray
    w: int
    h: int
    ref_x: np.ndarray
    ref_y: np.ndarray
    valid: np.ndarray


@dataclass
class _FakeCfg:
    pixel_size_m: float = 5.5e-6
    focal_length_m: float = 5.2e-3

    @property
    def slope_scale(self):
        return self.pixel_size_m / self.focal_length_m


def _build_frame_and_grid(shifts):
    """Render a 20x20 frame with 4 spots (2x2 cells, 10x10 windows) shifted from
    their reference centroids by ``shifts`` and return (frame, grid)."""
    frame = np.zeros((20, 20), dtype=np.float64)
    refs = [(4.5, 4.5), (14.5, 4.5), (4.5, 14.5), (14.5, 14.5)]
    cols = np.arange(20)[None, :]
    rows = np.arange(20)[:, None]
    for (rx, ry), (dx, dy) in zip(refs, shifts):
        frame += 1000.0 * np.exp(-((cols - (rx + dx)) ** 2
                                   + (rows - (ry + dy)) ** 2) / (2 * 1.8 ** 2))
    grid = _FakeGrid(
        x0=np.array([0., 10., 0., 10.]),
        y0=np.array([0., 0., 10., 10.]),
        w=10, h=10,
        ref_x=np.array([4.5, 14.5, 4.5, 14.5]),
        ref_y=np.array([4.5, 4.5, 14.5, 14.5]),
        valid=np.array([True, True, True, True]),
    )
    return frame, grid


def test_driver_centroid_frame_recovers_spots():
    """centroid_frame() returns absolute centroids per sub-aperture matching the
    injected spot positions."""
    shifts = [(0.3, -0.2), (-0.4, 0.1), (0.0, 0.5), (0.6, 0.6)]
    frame, grid = _build_frame_and_grid(shifts)
    cents = C.centroid_frame(frame, grid, method="twcog",
                             thresh_frac=0.05, fwhm_px=4.2)
    refs = [(4.5, 4.5), (14.5, 4.5), (4.5, 14.5), (14.5, 14.5)]
    for k, ((rx, ry), (dx, dy)) in enumerate(zip(refs, shifts)):
        assert abs(cents[k, 0] - (rx + dx)) < 0.02
        assert abs(cents[k, 1] - (ry + dy)) < 0.02


def test_driver_slope_vector_ordering_and_scale():
    """centroids_to_slopes() produces the canonical [sx_1..sx_M, sy_1..sy_M]
    vector with the correct pixel->rad scale and reference subtraction
    (ARCHITECTURE.md S3.2/S4.2)."""
    shifts = [(0.3, -0.2), (-0.4, 0.1), (0.2, 0.5), (0.6, 0.6)]
    frame, grid = _build_frame_and_grid(shifts)
    cfg = _FakeCfg()

    slopes = C.frame_to_slopes(frame, grid, cfg, method="twcog",
                               thresh_frac=0.05, fwhm_px=4.2)
    assert slopes.shape == (8,)   # 2 * 4 valid sub-apertures

    scale = cfg.slope_scale
    expected = np.array([dx for dx, dy in shifts] +
                        [dy for dx, dy in shifts]) * scale
    assert np.max(np.abs(slopes - expected)) < 5e-4 * scale + 2e-4


def test_driver_handles_dead_subaperture():
    """A dead sub-aperture centroids to NaN; centroids_to_slopes(fill_dead=...)
    emits zero slope (reference) so the vector has no NaNs for the MVM."""
    shifts = [(0.3, -0.2), (-0.4, 0.1), (0.2, 0.5), (0.6, 0.6)]
    frame, grid = _build_frame_and_grid(shifts)
    frame[10:20, 10:20] = 0.0   # kill the 4th cell

    cents = C.centroid_frame(frame, grid, method="twcog", thresh_frac=0.05)
    assert np.isnan(cents[3, 0]) and np.isnan(cents[3, 1])

    slopes = C.centroids_to_slopes(cents, grid, _FakeCfg(),
                                   fill_dead="reference")
    assert not np.any(np.isnan(slopes))
    # The dead cell contributes zero slope on both axes (index 3 is sx4, 7 is sy4).
    assert slopes[3] == 0.0 and slopes[7] == 0.0

    # fill_dead='nan' should preserve the NaN.
    slopes_nan = C.centroids_to_slopes(cents, grid, _FakeCfg(), fill_dead="nan")
    assert np.isnan(slopes_nan[3]) and np.isnan(slopes_nan[7])


def test_driver_skips_invalid_subapertures_in_vector():
    """Invalid (masked-off) sub-apertures are excluded from the slope vector,
    which has length 2 * n_valid."""
    shifts = [(0.3, -0.2), (-0.4, 0.1), (0.2, 0.5), (0.6, 0.6)]
    frame, grid = _build_frame_and_grid(shifts)
    grid.valid = np.array([True, True, False, True])   # 3 valid

    slopes = C.frame_to_slopes(frame, grid, _FakeCfg(), method="twcog",
                               thresh_frac=0.05)
    assert slopes.shape == (6,)   # 2 * 3 valid


def test_unknown_method_raises():
    """An unknown method name is rejected with a clear error."""
    _, grid = _build_frame_and_grid([(0, 0)] * 4)
    frame = np.zeros((20, 20))
    with pytest.raises(ValueError):
        C.centroid_frame(frame, grid, method="not_a_method")
