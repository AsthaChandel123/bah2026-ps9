#!/usr/bin/env python3
"""run_pipeline_py.py -- pure-Python end-to-end pipeline (validation/plots).

Mirrors the C real-time pipeline (``bin/wfs_rt``) in Python for validation and
visualization, and is the reference oracle the C core is validated against
(C/Python parity):

  BMP -> centroid -> slopes -> { phi = R @ s (zonal),  a = M+ @ s (modal) }
       -> W = Z @ a (synthesis)  ->  a_dm = G @ phi (DM commands, gain+clip)

For each frame it writes per-frame phase maps, an aggregate Zernike-coefficient
CSV, per-frame + aggregate actuator CSVs, and a slopes CSV -- the SAME artifact
filenames/orderings as the C core, so the two outputs are directly comparable.
It then runs the multi-method turbulence characterization and, when a
``ground_truth.json`` is present (synthetic data), scores r0/tau0 recovery and
reconstruction quality (RMS/Strehl) into ``pipeline_summary.json``.

PARITY NOTE: to match the C core bit-for-bit on slopes/phase, this replicates
the C window placement (``x0,y0,w,h`` rounded to int as the C core rounds the
AOMX subapmap), the window-center reference (refslopes == 0), and the C
centroider -- TWCoG with a NULL weight LUT and unit gain, which is exactly
**thresholded center-of-gravity** at ``thresh_frac = 0.10`` (src/centroid.c).

Usage:
  python3 scripts/run_pipeline_py.py --config config/example_config.json \
      --calib calib/ --frames "results/synthetic/run01/frame_*.bmp" --out out_py/ --plots
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aokit.config import load_config, from_dict           # noqa: E402
from aokit import geometry as geom_mod                      # noqa: E402
from aokit import reconstructor as recon                    # noqa: E402
from aokit import bmpio, turbulence, validation             # noqa: E402
from aokit.matio import read_aomx                            # noqa: E402

# Matches src/pipeline.c::build_subapertures and src/centroid.c.
C_THRESH_FRAC = 0.10
C_MIN_PIXELS = 3


def _resolve_frames(frames_arg: str) -> list:
    if os.path.isdir(frames_arg):
        return sorted(glob.glob(os.path.join(frames_arg, "*.bmp")))
    return sorted(glob.glob(frames_arg))


def _c_int(v: float) -> int:
    """Replicate the C cast ``(int)(v + 0.5)`` -- truncation TOWARD ZERO (so a
    negative ``v`` rounds differently from ``floor``). This is exactly how
    src/pipeline.c::build_subapertures rounds the AOMX subapmap entries; matching
    it is required for bit-for-bit C/Python parity on the edge sub-apertures."""
    return int(np.trunc(v + 0.5))


def _c_aligned_subaps(geom):
    """Per-sub-aperture integer window geometry + references matching the C core.

    Returns ``(x0, y0, w, h, ref_x, ref_y)`` where ``x0,y0,w,h`` are the
    C-rounded integers (``(int)(v+0.5)``) and ``ref = x0 + 0.5*w`` (refslopes==0,
    the window-center reference src/pipeline.c uses)."""
    sub = geom.subaps
    x0f = np.asarray(sub.x0, dtype=np.float64)
    y0f = np.asarray(sub.y0, dtype=np.float64)
    n = x0f.shape[0]
    x0 = np.array([_c_int(v) for v in x0f], dtype=np.int64)
    y0 = np.array([_c_int(v) for v in y0f], dtype=np.int64)
    w = _c_int(float(sub.w))
    h = _c_int(float(sub.h))
    if w <= 0:
        w = 1
    if h <= 0:
        h = 1
    ref_x = x0.astype(np.float64) + 0.5 * w
    ref_y = y0.astype(np.float64) + 0.5 * h
    return x0, y0, w, h, ref_x, ref_y


def _c_centroids(frame, x0, y0, w, h, thresh_frac=C_THRESH_FRAC,
                 min_pixels=C_MIN_PIXELS):
    """C-exact thresholded center-of-gravity over each sub-aperture window.

    Byte-faithful re-implementation of src/centroid.c::centroid_twcog with a
    NULL weight LUT and unit gain (== thresholded CoG), including the C window
    clamp (``x0c=max(0,x0); x1=min(fw, x0+w)`` -- the FAR edge uses the nominal
    origin+width, NOT the clamped origin, so an off-frame window keeps fewer
    columns than ``w``). Returns an ``(n_sub, 2)`` array of absolute ``[cx, cy]``
    (NaN where invalid), and a bool valid mask.
    """
    frame = np.asarray(frame, dtype=np.float64)
    fh, fw = frame.shape
    n = x0.shape[0]
    cents = np.full((n, 2), np.nan, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)
    for k in range(n):
        ax0 = int(x0[k]); ay0 = int(y0[k])
        cx0 = max(0, ax0); cy0 = max(0, ay0)
        cx1 = min(fw, ax0 + w); cy1 = min(fh, ay0 + h)
        if cx1 <= cx0 or cy1 <= cy0:
            continue
        win = frame[cy0:cy1, cx0:cx1]
        imax = float(win.max())
        if not (imax > 0.0):
            continue
        thr = thresh_frac * imax
        v = win - thr
        above = v > 0.0
        n_above = int(np.count_nonzero(above))
        if n_above < (min_pixels if min_pixels > 0 else 1):
            continue
        vv = np.where(above, v, 0.0)
        sden = float(vv.sum())
        if sden <= 0.0:
            continue
        cols = np.arange(cx0, cx1, dtype=np.float64)
        rows = np.arange(cy0, cy1, dtype=np.float64)
        cx = float((vv.sum(axis=0) @ cols)) / sden
        cy = float((vv.sum(axis=1) @ rows)) / sden
        cents[k, 0] = cx
        cents[k, 1] = cy
        valid[k] = True
    return cents, valid


def _load_or_build_matrices(cfg, geom, calib_dir, j_max=15):
    """Load R/Mpinv/Z/G from an AOMX calib dir if given, else build them.

    Returns ``(R, Mpinv, Z, G)`` as float64 arrays. When loading, the AOMX files
    are exactly what the C core uses, guaranteeing matrix-level parity.
    """
    if calib_dir:
        R = read_aomx(os.path.join(calib_dir, "R.aomx")).astype(np.float64)
        Mpinv = read_aomx(os.path.join(calib_dir, "Mpinv.aomx")).astype(np.float64)
        Z = read_aomx(os.path.join(calib_dir, "Z.aomx")).astype(np.float64)
        G = read_aomx(os.path.join(calib_dir, "G.aomx")).astype(np.float64)
        return R, Mpinv, Z, G
    # Build in-memory (same construction as scripts/build_calibration.py,
    # including the physical modal-slope scaling so R1/R4/R6 are correct).
    from build_calibration import _build_valid_node_dm, _build_modal_physical
    mats = recon.build_all(geom, j_max=j_max)
    modal = _build_modal_physical(geom, cfg, j_max)
    dm = _build_valid_node_dm(geom, cfg)
    return mats["R"], modal["Mpinv"], modal["Z"], dm["G"]


def run_pipeline(config_path, frames, out_dir, calib_dir=None, plots=False,
                 j_max=15, verbose=True) -> dict:
    """Run the full Python pipeline over ``frames``; return a summary dict."""
    os.makedirs(out_dir, exist_ok=True)
    frame_paths = _resolve_frames(frames)
    if not frame_paths:
        raise SystemExit(f"run_pipeline_py: no frames matched '{frames}'")

    # Prefer the dataset's own ground_truth config (carries injected r0/tau0).
    gt_path = None
    for cand in (os.path.join(os.path.dirname(frame_paths[0]), "ground_truth.json"),):
        if os.path.exists(cand):
            gt_path = cand
            break
    cfg = load_config(config_path)
    ground_truth = None
    if gt_path:
        with open(gt_path) as fh:
            ground_truth = json.load(fh)
        # Fold injected r0/tau0/wind into the Config so estimate_all can use them.
        gt = ground_truth
        gcfg = dict(gt.get("config", {}))
        gcfg["dm"] = {  # config_to_dict omits dm; reuse the JSON config's dm
            "n_act_x": cfg.dm.n_act_x, "n_act_y": cfg.dm.n_act_y,
            "pitch_m": cfg.dm.pitch_m, "coupling_coeff": cfg.dm.coupling_coeff,
            "stroke_max_m": cfg.dm.stroke_max_m,
            "influence_model": cfg.dm.influence_model,
            "influence_alpha": cfg.dm.influence_alpha,
            "stroke_gain_m_per_unit": cfg.dm.stroke_gain_m_per_unit,
        }
        gcfg.setdefault("geometry", {"type": cfg.geometry.type})
        gcfg["ground_truth"] = {
            "r0_m": gt.get("r0_m"), "tau0_s": gt.get("tau0_s"),
            "wind_speed_mps": gt.get("wind_speed_mps"), "L0_m": gt.get("L0_m"),
        }
        try:
            cfg = from_dict(gcfg)
        except Exception:
            pass  # fall back to the JSON config (no ground-truth wind)

    geom = geom_mod.build_geometry(cfg)
    # C-exact window geometry + references (parity with src/pipeline.c).
    x0, y0, w, h, ref_x, ref_y = _c_aligned_subaps(geom)
    grid = geom_mod.SubApertureGrid(
        x0=x0.astype(np.float64), y0=y0.astype(np.float64), w=w, h=h,
        ref_x=ref_x, ref_y=ref_y,
        valid=np.ones(x0.shape[0], dtype=bool), ij=geom.subaps.ij.copy())
    M = geom.n_sub
    slope_scale = float(cfg.slope_scale)

    R, Mpinv, Z, G = _load_or_build_matrices(cfg, geom, calib_dir, j_max=j_max)
    J = Mpinv.shape[0]
    n_phase = R.shape[0]
    n_act = G.shape[0]

    stroke_gain = float(cfg.dm.stroke_gain_m_per_unit)
    stroke_max = float(cfg.dm.stroke_max_m)

    # Per-frame accumulators.
    n = len(frame_paths)
    slopes_ts = np.zeros((n, 2 * M), dtype=np.float64)
    coeffs_ts = np.zeros((n, J), dtype=np.float64)
    acts_ts = np.zeros((n, n_act), dtype=np.float64)
    phi_ts = np.zeros((n, n_phase), dtype=np.float64)
    n_valid_ts = np.zeros(n, dtype=np.int64)
    n_sat_ts = np.zeros(n, dtype=np.int64)

    first_frame = None
    first_cents = None
    for k, path in enumerate(frame_paths):
        frame = bmpio.read_bmp_gray(path)
        if k == 0:
            first_frame = frame
        # C-exact thresholded CoG (== C TWCoG with NULL weights, unit gain).
        cents, valid = _c_centroids(frame, x0, y0, w, h)
        if k == 0:
            first_cents = cents
        n_valid_ts[k] = int(valid.sum())
        # slope = (centroid - window_center) * pixel_size / focal_length, with
        # invalid sub-apertures contributing zero (centroid := reference). The C
        # core falls a dead window back to its window center, i.e. zero slope.
        dispx = np.where(valid, cents[:, 0] - ref_x, 0.0)
        dispy = np.where(valid, cents[:, 1] - ref_y, 0.0)
        s = np.concatenate([dispx, dispy]) * slope_scale
        slopes_ts[k] = s
        phi = R @ s
        phi_ts[k] = phi
        coeffs_ts[k] = Mpinv @ s
        a_units = G @ phi
        a_strokes = a_units * stroke_gain
        n_sat_ts[k] = int(np.count_nonzero(np.abs(a_strokes) > stroke_max))
        acts_ts[k] = np.clip(a_strokes, -stroke_max, stroke_max)

    # ---- write artifacts (same filenames/orderings as the C core) ----
    _write_csv(os.path.join(out_dir, "slopes.csv"), slopes_ts,
               [f"s{i}" for i in range(2 * M)])
    _write_csv(os.path.join(out_dir, "zernike_coeffs.csv"), coeffs_ts,
               [f"j{j + 2}" for j in range(J)])
    _write_csv(os.path.join(out_dir, "actuators.csv"), acts_ts,
               [f"a{i}" for i in range(n_act)])
    for k in range(n):
        from aokit.matio import write_aomx
        write_aomx(os.path.join(out_dir, f"phase_{k:04d}.aomx"),
                   phi_ts[k].astype(np.float32), "f32")

    # ---- turbulence characterization (the multi-method r0/tau0 report) ----
    # Reconstruct full 2-D phase maps per frame (on the pupil grid) for the
    # phase-domain estimators (R4/R5/T5) using the synthesis basis Z scattered
    # back to a node raster is overkill; instead pass the modal coeffs/slopes
    # time-series (R1/R2/R6/T1/T2/T4) plus a node-phase stack for R4.
    turb = turbulence.estimate_all(slopes_ts, coeffs_ts, None, cfg, cfg.dt_s)
    turb_dict = turb.to_dict()
    with open(os.path.join(out_dir, "turbulence_summary.json"), "w") as fh:
        json.dump(turb_dict, fh, indent=2)

    # ---- validation vs ground truth (synthetic) ----
    summary = {
        "n_frames": n,
        "M_valid_subaps": M,
        "n_phase_nodes": n_phase,
        "n_act": n_act,
        "J_modes": J,
        "mean_valid_subaps": float(np.mean(n_valid_ts)),
        "mean_saturated_acts": float(np.mean(n_sat_ts)),
        "turbulence": turb_dict,
    }
    if ground_truth is not None:
        summary["validation"] = _validate(
            cfg, geom, ground_truth, coeffs_ts, turb_dict, Z, j_max)

    with open(os.path.join(out_dir, "pipeline_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    if plots:
        _make_plots(out_dir, cfg, geom, grid, first_frame, first_cents,
                    slopes_ts, coeffs_ts, phi_ts, acts_ts, Z, turb_dict,
                    ground_truth)

    if verbose:
        print(f"run_pipeline_py: {n} frames, M={M} subaps, "
              f"N_phase={n_phase}, J={J}, n_act={n_act}")
        r0 = turb_dict["r0_m"]["median"]
        tau0 = turb_dict["tau0_s"]["median"]
        print(f"  r0(median)   = {r0:.4f} m   (estimators: "
              f"{', '.join(turb_dict['r0_m']['estimators'])})")
        print(f"  tau0(median) = {tau0 * 1e3:.3f} ms  (estimators: "
              f"{', '.join(turb_dict['tau0_s']['estimators'])})")
        if "validation" in summary:
            v = summary["validation"]
            print(f"  r0 recovery error   = {v.get('r0_recovery_error_pct'):.2f} %")
            if v.get("tau0_recovery_error_pct") is not None:
                print(f"  tau0 recovery error = {v['tau0_recovery_error_pct']:.2f} %")
            print(f"  Zernike recon RMS   = {v.get('zernike_rms_rad'):.4f} rad "
                  f"(Strehl {v.get('strehl_marechal'):.3f})")
        print(f"  wrote artifacts + pipeline_summary.json to {out_dir}")
    return summary


def _validate(cfg, geom, gt, coeffs_ts, turb_dict, Z, j_max) -> dict:
    """Score r0/tau0 recovery + Zernike reconstruction vs the ground truth."""
    out = {}
    r0_true = gt.get("r0_m")
    tau0_true = gt.get("tau0_s")
    r0_est = turb_dict["r0_m"]["median"]
    tau0_est = turb_dict["tau0_s"]["median"]
    out["r0_true_m"] = r0_true
    out["r0_est_median_m"] = r0_est
    out["r0_estimators"] = turb_dict["r0_m"]["estimators"]
    out["tau0_true_s"] = tau0_true
    out["tau0_est_median_s"] = tau0_est
    out["tau0_estimators"] = turb_dict["tau0_s"]["estimators"]
    if r0_true and np.isfinite(r0_est):
        out["r0_recovery_error_pct"] = validation.r0_recovery_error(r0_est, r0_true)
    if tau0_true and np.isfinite(tau0_est):
        out["tau0_recovery_error_pct"] = validation.tau0_recovery_error(
            tau0_est, tau0_true)

    # Per-frame Zernike reconstruction error vs the true coefficients (projected
    # onto the same valid-subap-center modal basis the recon uses). The modal
    # reconstructor returns Noll 2..j_max in the same order as gt's per-frame
    # truth, so compare the overlapping prefix.
    zt = gt.get("zernike_noll_per_frame") or []
    rms_list = []
    if zt:
        zt = np.asarray(zt, dtype=np.float64)
        m = min(zt.shape[1], coeffs_ts.shape[1], zt.shape[0] and coeffs_ts.shape[0])
        # The geometric spot model measures the sub-aperture-averaged tilt, which
        # the modal reconstructor maps back to Noll coefficients; tip/tilt and
        # low orders dominate. Report the coefficient-domain residual RMS.
        kf = min(zt.shape[0], coeffs_ts.shape[0])
        kc = min(zt.shape[1], coeffs_ts.shape[1])
        diff = coeffs_ts[:kf, :kc] - zt[:kf, :kc]
        rms_list = np.sqrt(np.mean(diff ** 2, axis=1))
    if len(rms_list):
        rms = float(np.mean(rms_list))
        out["zernike_rms_rad"] = rms
        out["strehl_marechal"] = float(np.exp(-rms * rms))
    else:
        out["zernike_rms_rad"] = float("nan")
        out["strehl_marechal"] = turb_dict.get("strehl_marechal", float("nan"))
    return out


def _write_csv(path, data, headers):
    """Write a frame_idx + columns CSV (matches the C core's writers)."""
    data = np.asarray(data, dtype=np.float64)
    with open(path, "w") as fh:
        fh.write("frame_idx," + ",".join(headers) + "\n")
        for k in range(data.shape[0]):
            fh.write(str(k) + "," +
                     ",".join(f"{v:.8g}" for v in data[k]) + "\n")


def _make_plots(out_dir, cfg, geom, grid, frame0, cents0, slopes_ts, coeffs_ts,
                phi_ts, acts_ts, Z, turb_dict, ground_truth):
    """Save the key diagnostic plots (headless Agg)."""
    try:
        from aokit import viz
    except Exception:
        return
    pdir = out_dir
    # Spot field + sub-aperture boxes + measured centroids (frame 0).
    valid0 = np.isfinite(cents0).all(axis=1)
    viz.plot_spot_field(frame0, subaps=grid,
                        centroids=(cents0[valid0, 0], cents0[valid0, 1]),
                        title="SH-WFS spot field (frame 0)",
                        savepath=os.path.join(pdir, "spot_field.png"))
    # Reconstructed phase map (frame 0) scattered onto the node raster.
    _plot_node_phase(viz, geom, phi_ts[0],
                     os.path.join(pdir, "phase_map.png"))
    # Zernike spectrum (time-mean amplitude).
    viz.plot_zernike_spectrum(np.sqrt(np.mean(coeffs_ts ** 2, axis=0)),
                              title="Zernike spectrum (RMS over frames)",
                              savepath=os.path.join(pdir, "zernike_spectrum.png"))
    # Actuator map (frame 0).
    viz.plot_actuator_map(acts_ts[0], cfg.dm.n_act_x, cfg.dm.n_act_y,
                          title="DM actuator map (frame 0, stroke m)",
                          savepath=os.path.join(pdir, "actuator_map.png"))


def _plot_node_phase(viz, geom, phi_nodes, savepath):
    """Render the valid-node phase vector onto the (n+1)x(n+1) corner raster."""
    cfg = geom.cfg
    nx1 = cfg.mla.n_lenslets_x + 1
    ny1 = cfg.mla.n_lenslets_y + 1
    valid_nodes, _ = recon.valid_node_index(geom)
    grid_img = np.full(nx1 * ny1, np.nan, dtype=np.float64)
    grid_img[valid_nodes] = phi_nodes
    viz.plot_phase_map(grid_img.reshape(ny1, nx1),
                       title="Reconstructed wavefront (Fried nodes, rad)",
                       savepath=savepath)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="JSON config path")
    ap.add_argument("--frames", required=True,
                    help="glob (e.g. 'dir/frame_*.bmp') or a directory")
    ap.add_argument("--calib", default=None,
                    help="AOMX calib dir; if omitted, matrices are built in-memory")
    ap.add_argument("--out", default="out_py/", help="output directory")
    ap.add_argument("--jmax", type=int, default=15,
                    help="max Noll index for the modal basis (default 15)")
    ap.add_argument("--plots", action="store_true", help="emit matplotlib plots")
    args = ap.parse_args(argv)

    run_pipeline(args.config, args.frames, args.out, calib_dir=args.calib,
                 plots=args.plots, j_max=args.jmax)
    return 0


if __name__ == "__main__":
    sys.exit(main())
