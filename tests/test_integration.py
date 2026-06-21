"""End-to-end integration tests for the PS9 pipeline.

Exercises the full inject-r0/tau0 -> recover loop on a TINY, fixed-seed case
(fast, < ~20 s):

    generate_dataset  ->  build_calibration  ->  run_pipeline_py
      -> assert the multi-method combiner recovers the injected r0 within
         tolerance and that the reconstruction quality (Strehl / Zernike RMS)
         is reasonable;
      -> assert C-vs-Python parity by invoking bin/wfs_rt over the same frames
         (skipped gracefully when the binary is not built).

The scripts live in ``scripts/`` and import ``aokit``; we add both to sys.path.
A small (6x6 lenslet, 128x128) configuration with a sub-mm pupil keeps the
turbulence well-sampled and the run sub-second.  The injected r0 (0.6 mm over a
0.9 mm pupil, D/r0 = 1.5) is in the regime where the variance estimators are
well-conditioned; tolerances are deliberately generous because turbulence
statistics over a small aperture / short sequence carry real realization scatter
(the combiner's spread reports this honestly).
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(REPO, "scripts")
for p in (REPO, SCRIPTS):
    if p not in sys.path:
        sys.path.insert(0, p)

# Fixed tiny configuration (sub-mm pupil so D/r0 is meaningful at r0 ~ 0.6 mm).
TINY_CONFIG = {
    "schema_version": 1,
    "camera": {"pixel_size_m": 5.5e-6, "frame_w": 128, "frame_h": 128,
               "bit_depth": 8},
    "mla": {"n_lenslets_x": 6, "n_lenslets_y": 6, "pitch_m": 1.5e-4,
            "focal_length_m": 5.2e-3},
    "pupil": {"diameter_m": 9.0e-4, "center_x_px": 64.0, "center_y_px": 64.0},
    "wavelength_m": 6.33e-7,
    "dm": {"n_act_x": 7, "n_act_y": 7, "pitch_m": 1.5e-4, "coupling_coeff": 0.15,
           "stroke_max_m": 3.5e-6, "influence_model": "gaussian",
           "influence_alpha": 2.0, "stroke_gain_m_per_unit": 1.0e-6},
    "geometry": {"type": "fried", "rotation_deg": 0.0, "flip_y": False},
    "cadence": {"dt_s": 2.0e-3},
}

R0_TRUE = 0.0006     # injected Fried parameter (m); D/r0 = 1.5
TAU0_TRUE = 0.005    # injected coherence time (s)
N_FRAMES = 14
SEED = 99


@pytest.fixture(scope="module")
def pipeline_run(tmp_path_factory):
    """Generate -> calibrate -> Python pipeline once; return the paths + summary."""
    from build_calibration import build_calibration
    from run_pipeline_py import run_pipeline
    from aokit.config import load_config
    from aokit import datagen

    d = tmp_path_factory.mktemp("e2e")
    cfg_path = os.path.join(d, "tiny_config.json")
    with open(cfg_path, "w") as fh:
        json.dump(TINY_CONFIG, fh)

    calib = os.path.join(d, "calib")
    data = os.path.join(d, "data")
    outpy = os.path.join(d, "py")

    shapes = build_calibration(cfg_path, calib, verbose=False)
    cfg = load_config(cfg_path)
    datagen.generate_dataset(
        cfg, r0_m=R0_TRUE, tau0_s=TAU0_TRUE, n_frames=N_FRAMES, out_dir=data,
        spot_model="geometric", L0_m=25.0, j_max=15, seed=SEED)
    summary = run_pipeline(cfg_path, os.path.join(data, "frame_*.bmp"), outpy,
                           calib_dir=calib, plots=False, verbose=False)
    return {
        "dir": str(d), "config": cfg_path, "calib": calib, "data": data,
        "outpy": outpy, "summary": summary, "shapes": shapes,
    }


def test_calibration_shapes_consistent(pipeline_run):
    """R (N_phase x 2M), G (N_phase x N_phase), Z (N_phase x J) line up so the
    C pipeline phi = R@s ; a = G@phi is dimensionally consistent."""
    s = pipeline_run["shapes"]
    n_phase = s["n_phase"]
    two_m = s["two_m"]
    J = s["J"]
    assert s["R"] == (n_phase, two_m)
    assert s["Z"] == (n_phase, J)
    assert s["G"] == (n_phase, n_phase)      # G consumes phi (length N_phase)
    assert s["Mpinv"] == (J, two_m)
    assert s["subapmap"][1] == 4             # [x0, y0, w, h]
    assert s["refslopes"] == (two_m,)


def test_ground_truth_written(pipeline_run):
    gt = os.path.join(pipeline_run["data"], "ground_truth.json")
    assert os.path.exists(gt)
    with open(gt) as fh:
        d = json.load(fh)
    assert abs(d["r0_m"] - R0_TRUE) < 1e-12
    assert abs(d["tau0_s"] - TAU0_TRUE) < 1e-9
    assert len(d["frames"]) == N_FRAMES
    assert len(d["zernike_noll_per_frame"]) == N_FRAMES


def test_multimethod_r0_recovery(pipeline_run):
    """The multi-method combiner recovers the injected r0 within tolerance and
    runs the full estimator bank (>=5 r0, >=6 tau0 methods)."""
    v = pipeline_run["summary"]["validation"]
    turb = pipeline_run["summary"]["turbulence"]
    assert len(turb["r0_m"]["estimators"]) >= 5
    assert len(turb["tau0_s"]["estimators"]) >= 6

    r0_est = v["r0_est_median_m"]
    assert np.isfinite(r0_est)
    # Generous tolerance (small-aperture realization scatter): within ~35% and
    # within a factor of 2 of the injected value.
    assert v["r0_recovery_error_pct"] < 35.0
    assert 0.5 * R0_TRUE < r0_est < 2.0 * R0_TRUE


def test_tau0_recovery(pipeline_run):
    """tau0 is recovered to within the documented (loose) coherence-time band."""
    v = pipeline_run["summary"]["validation"]
    tau0_est = v["tau0_est_median_s"]
    assert np.isfinite(tau0_est)
    # tau0 estimators are inherently softer; accept within ~60% and a factor 3.
    assert v["tau0_recovery_error_pct"] < 60.0
    assert TAU0_TRUE / 3.0 < tau0_est < TAU0_TRUE * 3.0


def test_reconstruction_quality(pipeline_run):
    """Reconstruction quality is sane: finite Zernike RMS and Strehl in (0, 1]."""
    v = pipeline_run["summary"]["validation"]
    rms = v["zernike_rms_rad"]
    strehl = v["strehl_marechal"]
    assert np.isfinite(rms) and rms >= 0.0
    assert 0.0 < strehl <= 1.0
    # The whole valid sub-aperture set should stay illuminated for this geometry.
    assert pipeline_run["summary"]["mean_valid_subaps"] > 0


def test_artifacts_written(pipeline_run):
    outpy = pipeline_run["outpy"]
    for name in ("slopes.csv", "zernike_coeffs.csv", "actuators.csv",
                 "pipeline_summary.json", "turbulence_summary.json"):
        assert os.path.exists(os.path.join(outpy, name)), name
    # one phase map per frame
    assert len(glob.glob(os.path.join(outpy, "phase_*.aomx"))) == N_FRAMES


@pytest.mark.skipif(not os.path.exists(os.path.join(REPO, "bin", "wfs_rt")),
                    reason="bin/wfs_rt not built (run `make` first)")
def test_c_vs_python_parity(pipeline_run):
    """The optimized C core and the Python reference agree to float32 precision
    on slopes and Zernike coefficients over the same frames."""
    binp = os.path.join(REPO, "bin", "wfs_rt")
    calib = pipeline_run["calib"]
    data = pipeline_run["data"]
    outc = os.path.join(pipeline_run["dir"], "c")
    frames = sorted(glob.glob(os.path.join(data, "frame_*.bmp")))

    r = subprocess.run(
        [binp, "--config", os.path.join(calib, "config.txt"),
         "--calib", calib, "--frames", *frames, "--out", outc],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr

    sc = _load_csv(os.path.join(outc, "slopes.csv"))
    sp = _load_csv(os.path.join(pipeline_run["outpy"], "slopes.csv"))
    zc = _load_csv(os.path.join(outc, "zernike_coeffs.csv"))
    zp = _load_csv(os.path.join(pipeline_run["outpy"], "zernike_coeffs.csv"))

    assert sc.shape == sp.shape and zc.shape == zp.shape
    # float32 matrices + float32 C arithmetic vs float64 Python: ~1e-5 abs is the
    # round-off floor. Slopes are ~1e-3 rad; coeffs ~1 rad.
    assert np.max(np.abs(sc - sp)) < 1e-4
    assert np.max(np.abs(zc - zp)) < 1e-3


def _load_csv(path):
    """Load a frame_idx,c0,c1,... CSV, dropping the frame_idx column."""
    rows = []
    with open(path) as fh:
        fh.readline()
        for line in fh:
            line = line.strip()
            if line:
                rows.append([float(x) for x in line.split(",")[1:]])
    return np.asarray(rows, dtype=np.float64)
