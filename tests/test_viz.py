"""Headless smoke tests for aokit.viz.

Each plotting function is called on small synthetic arrays with a temporary
``savepath``; the test asserts the PNG was written and is non-empty, then closes
all figures.  No display is required -- the module forces the Agg backend at
import time, and we belt-and-suspenders it here too.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from aokit import viz  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figs():
    """Close any leftover figures after every test to avoid warnings."""
    yield
    plt.close("all")


def _assert_png(path):
    assert path.exists(), f"expected PNG at {path}"
    assert path.stat().st_size > 0, f"PNG at {path} is empty"


def test_backend_is_agg():
    assert matplotlib.get_backend().lower() == "agg"


def test_plot_spot_field(tmp_path):
    rng = np.random.default_rng(0)
    frame = rng.random((32, 32))
    p = tmp_path / "spot_field.png"
    fig = viz.plot_spot_field(frame, savepath=str(p), title="t")
    _assert_png(p)
    plt.close(fig)


def test_plot_spot_field_with_grid(tmp_path):
    # Minimal SubApertureGrid-like object (duck-typed).
    class G:
        x0 = np.array([2.0, 12.0, 2.0, 12.0])
        y0 = np.array([2.0, 2.0, 12.0, 12.0])
        w = 8
        h = 8
        ref_x = np.array([6.0, 16.0, 6.0, 16.0])
        ref_y = np.array([6.0, 6.0, 16.0, 16.0])
        valid = np.array([True, True, True, False])

    frame = np.zeros((28, 28))
    centroids = np.column_stack([G.ref_x + 0.5, G.ref_y - 0.5])
    p = tmp_path / "spot_field_grid.png"
    fig = viz.plot_spot_field(frame, subaps=G, centroids=centroids,
                              savepath=str(p))
    _assert_png(p)
    plt.close(fig)


def test_plot_slope_quiver_separate(tmp_path):
    x = np.array([0.0, 1.0, 0.0, 1.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    sx = np.array([0.1, -0.1, 0.05, 0.0])
    sy = np.array([0.0, 0.1, -0.05, 0.1])
    p = tmp_path / "quiver.png"
    fig = viz.plot_slope_quiver((x, y), sx, sy, savepath=str(p))
    _assert_png(p)
    plt.close(fig)


def test_plot_slope_quiver_block_vector(tmp_path):
    # Full 2M block vector [sx_1..sx_M, sy_1..sy_M], positions as (N, 2).
    pos = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    s = np.array([0.1, 0.2, -0.1, 0.0, 0.05, -0.05])
    p = tmp_path / "quiver_block.png"
    fig = viz.plot_slope_quiver(pos, s, savepath=str(p))
    _assert_png(p)
    plt.close(fig)


def test_plot_phase_map(tmp_path):
    lin = np.linspace(-1, 1, 24)
    xx, yy = np.meshgrid(lin, lin)
    W = xx ** 2 - yy ** 2
    mask = (xx ** 2 + yy ** 2) <= 1.0
    p = tmp_path / "phase.png"
    fig = viz.plot_phase_map(W, mask=mask, savepath=str(p), title="phase")
    _assert_png(p)
    plt.close(fig)


def test_plot_phase_map_from_vector(tmp_path):
    # A 1-D phase vector should be reshaped to a square image without error.
    W = np.linspace(-1, 1, 36)
    p = tmp_path / "phase_vec.png"
    fig = viz.plot_phase_map(W, savepath=str(p))
    _assert_png(p)
    plt.close(fig)


def test_plot_zernike_spectrum(tmp_path):
    coeffs = np.array([0.3, -0.2, 0.15, 0.05, -0.1, 0.02, 0.01, 0.03])
    p = tmp_path / "zernike.png"
    fig = viz.plot_zernike_spectrum(coeffs, savepath=str(p))
    _assert_png(p)
    plt.close(fig)


def test_plot_zernike_spectrum_kolmogorov_overlay(tmp_path):
    coeffs = np.linspace(0.3, 0.01, 14)
    p = tmp_path / "zernike_kolm.png"
    # Use the analytic Kolmogorov overlay via D/r0.
    fig = viz.plot_zernike_spectrum(coeffs, savepath=str(p), D_over_r0=10.0)
    _assert_png(p)
    plt.close(fig)


def test_plot_zernike_spectrum_array_overlay(tmp_path):
    coeffs = np.linspace(0.3, 0.01, 10)
    expect = np.linspace(0.25, 0.02, 10)
    p = tmp_path / "zernike_arr.png"
    fig = viz.plot_zernike_spectrum(coeffs, savepath=str(p), kolmogorov=expect)
    _assert_png(p)
    plt.close(fig)


def test_plot_r0_tau0_timeseries(tmp_path):
    t = np.arange(50) * 2e-3
    r0 = 0.15 + 0.01 * np.sin(t * 10)
    tau0 = 0.0045 + 0.0005 * np.cos(t * 10)
    p = tmp_path / "trends.png"
    fig = viz.plot_r0_tau0_timeseries(t, r0_series=r0, tau0_series=tau0,
                                      savepath=str(p))
    _assert_png(p)
    plt.close(fig)


def test_plot_r0_tau0_timeseries_r0_only(tmp_path):
    t = np.arange(20)
    r0 = 0.15 + 0.01 * np.random.default_rng(1).standard_normal(20)
    p = tmp_path / "trends_r0.png"
    fig = viz.plot_r0_tau0_timeseries(t, r0_series=r0, savepath=str(p))
    _assert_png(p)
    plt.close(fig)


def test_plot_turbulence_trends(tmp_path):
    r0 = np.full(30, 0.15)
    tau0 = np.full(30, 0.0045)
    p = tmp_path / "turb_trends.png"
    fig = viz.plot_turbulence_trends(r0, tau0, dt_s=2e-3, savepath=str(p))
    _assert_png(p)
    plt.close(fig)


def test_plot_residual(tmp_path):
    lin = np.linspace(-1, 1, 20)
    xx, yy = np.meshgrid(lin, lin)
    true = xx ** 2 - yy ** 2
    recon = true + 0.05 * np.random.default_rng(2).standard_normal(true.shape)
    mask = (xx ** 2 + yy ** 2) <= 1.0
    p = tmp_path / "residual.png"
    fig = viz.plot_residual(true, recon, mask=mask, savepath=str(p))
    _assert_png(p)
    plt.close(fig)


def test_plot_actuator_map(tmp_path):
    nx, ny = 6, 5
    cmd = np.random.default_rng(3).standard_normal(nx * ny)
    p = tmp_path / "actuators.png"
    fig = viz.plot_actuator_map(cmd, nx, ny, savepath=str(p))
    _assert_png(p)
    plt.close(fig)


def test_plot_method_comparison(tmp_path):
    labels = ["R1", "R2", "R3", "R4", "R5", "R6", "R7"]
    values = [0.149, 0.146, 0.150, 0.144, 0.151, 0.148, 0.145]
    errors = [0.003, 0.005, 0.002, 0.006, 0.004, 0.003, 0.005]
    p = tmp_path / "compare.png"
    fig = viz.plot_method_comparison(labels, values, errors=errors,
                                     reference=0.148, savepath=str(p))
    _assert_png(p)
    plt.close(fig)


def test_plot_method_comparison_no_errors(tmp_path):
    labels = ["T1", "T2", "T3"]
    values = [0.0045, 0.0047, 0.0046]
    p = tmp_path / "compare_tau.png"
    fig = viz.plot_method_comparison(labels, values, savepath=str(p),
                                     ylabel="tau0 (s)")
    _assert_png(p)
    plt.close(fig)


def test_summary_figure(tmp_path):
    lin = np.linspace(-1, 1, 20)
    xx, yy = np.meshgrid(lin, lin)
    phase = xx ** 2 - yy ** 2
    mask = (xx ** 2 + yy ** 2) <= 1.0
    coeffs = np.linspace(0.3, 0.01, 10)
    x = np.array([0.0, 1.0, 0.0, 1.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    sx = np.array([0.1, -0.1, 0.05, 0.0])
    sy = np.array([0.0, 0.1, -0.05, 0.1])
    t = np.arange(30) * 2e-3
    r0 = np.full(30, 0.15)
    tau0 = np.full(30, 0.0045)
    p = tmp_path / "summary.png"
    fig = viz.summary_figure(
        phase=phase, coeffs=coeffs,
        slopes_grid=((x, y), sx, sy),
        r0_tau0=(t, r0, tau0),
        mask=mask, savepath=str(p),
    )
    _assert_png(p)
    plt.close(fig)


def test_returns_figure_without_savepath(tmp_path):
    # Without a savepath the live Figure is returned (and we close it).
    fig = viz.plot_phase_map(np.ones((8, 8)))
    assert fig is not None
    plt.close(fig)


def test_ax_reuse_does_not_close(tmp_path):
    # Passing an ax should draw into it and return its parent figure.
    fig, ax = plt.subplots()
    out = viz.plot_zernike_spectrum(np.array([0.1, 0.2, 0.3]), ax=ax)
    assert out is fig
    assert len(ax.patches) > 0  # bars were drawn
    plt.close(fig)
