#!/usr/bin/env python3
"""analyze_turbulence.py -- multi-method r0/tau0 report from a time-series.

Loads a slopes / Zernike-coefficient time-series (as written by the C core or
``run_pipeline_py.py``) -- or computes one on the fly from a frame directory +
calibration -- then runs the full estimator bank (>=7 r0 estimators R1..R7,
>=6 tau0 estimators T1..T6 across slope/phase/intensity domains), removes
biases, reconciles them (per-method value + combined mean/std/median), and
writes ``turbulence_summary.json`` plus a cross-validation table and the
method-comparison bar chart. research/04 S5.

Inputs (pick one):
  --slopes  slopes.csv      (T x 2M, frame_idx + s0..s{2M-1})
  --coeffs  zernike_coeffs.csv (T x J, frame_idx + j2..jJ)
  --frames  DIR/glob + --calib DIR   (compute slopes/coeffs from frames)

Usage:
  python3 scripts/analyze_turbulence.py --slopes out/slopes.csv \
      --coeffs out/zernike_coeffs.csv --config config/example_config.json \
      --out out/turbulence_summary.json --plots
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aokit.config import load_config        # noqa: E402
from aokit import turbulence                 # noqa: E402


def _load_csv_timeseries(path: str) -> np.ndarray:
    """Load a frame_idx,c0,c1,... CSV, dropping the frame_idx column."""
    rows = []
    with open(path) as fh:
        header = fh.readline()  # skip header
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            rows.append([float(x) for x in parts[1:]])  # drop frame_idx
    if not rows:
        raise SystemExit(f"analyze_turbulence: empty time-series {path}")
    return np.asarray(rows, dtype=np.float64)


def _compute_from_frames(config_path, frames, calib_dir, j_max=15):
    """Compute slopes_ts/coeffs_ts by running the Python pipeline on frames."""
    from run_pipeline_py import run_pipeline  # reuse the pipeline
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        run_pipeline(config_path, frames, td, calib_dir=calib_dir,
                     plots=False, j_max=j_max, verbose=False)
        slopes = _load_csv_timeseries(os.path.join(td, "slopes.csv"))
        coeffs = _load_csv_timeseries(os.path.join(td, "zernike_coeffs.csv"))
    return slopes, coeffs


def _fmt(v, scale=1.0, unit=""):
    if v is None or not np.isfinite(v):
        return "    n/a "
    return f"{v * scale:8.4f}{unit}"


def _print_table(name, combo, scale, unit, true_val=None):
    """Print a per-method + combined cross-validation table."""
    print(f"\n  {name} estimators ({unit.strip() or 'value'}):")
    print("  " + "-" * 46)
    methods = [k for k in combo if k not in ("mean", "std", "median")]
    for m in methods:
        print(f"    {m:<22s} {_fmt(combo[m], scale, unit)}")
    print("  " + "-" * 46)
    print(f"    {'mean':<22s} {_fmt(combo['mean'], scale, unit)}")
    print(f"    {'std (spread)':<22s} {_fmt(combo['std'], scale, unit)}")
    print(f"    {'MEDIAN (combined)':<22s} {_fmt(combo['median'], scale, unit)}")
    if true_val is not None:
        print(f"    {'injected (truth)':<22s} {_fmt(true_val, scale, unit)}")
        if np.isfinite(combo["median"]) and true_val:
            err = 100.0 * abs(combo["median"] - true_val) / true_val
            print(f"    {'recovery error':<22s} {err:8.2f} %")


def _discover_ground_truth(*hints) -> dict | None:
    """Find a dataset ground_truth.json near any of the given path hints."""
    for h in hints:
        if not h:
            continue
        base = h
        if not os.path.isdir(base):
            base = os.path.dirname(os.path.abspath(base))
        cand = os.path.join(base, "ground_truth.json")
        if os.path.exists(cand):
            with open(cand) as fh:
                return json.load(fh)
    return None


def _cfg_with_truth(cfg, gt):
    """Return a Config whose ground_truth reflects the INJECTED r0/tau0/wind so
    estimate_all uses the right wind for the Greenwood/Tyler tau0 estimators."""
    if gt is None:
        return cfg
    from aokit.config import GroundTruth
    cfg.ground_truth = GroundTruth(
        r0_m=gt.get("r0_m"), tau0_s=gt.get("tau0_s"),
        wind_speed_mps=gt.get("wind_speed_mps"), L0_m=gt.get("L0_m"))
    return cfg


def analyze(config_path, slopes_path=None, coeffs_path=None, frames=None,
            calib_dir=None, out_path="turbulence_summary.json", plots=False,
            dt_s=None, j_max=15, ground_truth_path=None, verbose=True) -> dict:
    cfg = load_config(config_path)
    dt = float(dt_s) if dt_s is not None else float(cfg.dt_s)

    # Prefer the injected ground truth (from the dataset) over the (possibly
    # stale) ground_truth block baked into the JSON config.
    gt = _discover_ground_truth(ground_truth_path, frames, slopes_path,
                                coeffs_path)
    if gt is not None:
        cfg = _cfg_with_truth(cfg, gt)

    slopes_ts = coeffs_ts = None
    if frames:
        slopes_ts, coeffs_ts = _compute_from_frames(
            config_path, frames, calib_dir, j_max=j_max)
    else:
        if slopes_path:
            slopes_ts = _load_csv_timeseries(slopes_path)
        if coeffs_path:
            coeffs_ts = _load_csv_timeseries(coeffs_path)
    if slopes_ts is None and coeffs_ts is None:
        raise SystemExit("analyze_turbulence: provide --slopes/--coeffs or "
                         "--frames (+--calib)")

    # Full result (medians, spreads, derived seeing/Strehl).
    result = turbulence.estimate_all(slopes_ts, coeffs_ts, None, cfg, dt)
    res_dict = result.to_dict()

    # Cross-validation tables (per-method value + combined mean/std/median).
    r0_combo = turbulence.combine_r0(result.r0_m)
    tau0_combo = turbulence.combine_tau0(result.tau0_s)
    res_dict["r0_m"]["cross_validation"] = r0_combo
    res_dict["tau0_s"]["cross_validation"] = tau0_combo

    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(res_dict, fh, indent=2)

    if verbose:
        n = result.n_frames
        print(f"analyze_turbulence: {n} frames, dt={dt * 1e3:.3f} ms")
        print(f"  domains: slopes={'yes' if slopes_ts is not None else 'no'}, "
              f"coeffs={'yes' if coeffs_ts is not None else 'no'}")
        print(f"  r0 estimators run: {len(result.r0_m)}  "
              f"tau0 estimators run: {len(result.tau0_s)}")
        r0_true = cfg.ground_truth.r0_m if cfg.ground_truth else None
        tau0_true = cfg.ground_truth.tau0_s if cfg.ground_truth else None
        _print_table("r0", r0_combo, 1.0, " m", r0_true)
        _print_table("tau0", tau0_combo, 1e3, " ms", tau0_true)
        if result.seeing_arcsec is not None:
            print(f"\n  seeing  = {result.seeing_arcsec:.3f} arcsec")
        if result.strehl_marechal is not None:
            print(f"  Strehl  = {result.strehl_marechal:.3f} (Marechal, from r0)")
        if result.wind_speed_mps is not None:
            print(f"  wind    = {result.wind_speed_mps:.3f} m/s")
        print(f"\n  wrote {out_path}")

    if plots:
        _plot(out_dir, r0_combo, tau0_combo, cfg)

    return res_dict


def _plot(out_dir, r0_combo, tau0_combo, cfg):
    try:
        from aokit import viz
    except Exception:
        return
    r0_true = cfg.ground_truth.r0_m if cfg.ground_truth else None
    tau0_true = cfg.ground_truth.tau0_s if cfg.ground_truth else None

    r0_methods = [k for k in r0_combo if k not in ("mean", "std", "median")]
    r0_vals = [r0_combo[k] for k in r0_methods]
    viz.plot_method_comparison(
        r0_methods, r0_vals,
        title="r0 estimator cross-validation", ylabel="r0 (m)",
        reference=r0_true if r0_true else r0_combo["median"],
        savepath=os.path.join(out_dir, "r0_method_comparison.png"))

    tau0_methods = [k for k in tau0_combo if k not in ("mean", "std", "median")]
    tau0_vals = [tau0_combo[k] * 1e3 for k in tau0_methods]
    viz.plot_method_comparison(
        tau0_methods, tau0_vals,
        title="tau0 estimator cross-validation", ylabel="tau0 (ms)",
        reference=(tau0_true * 1e3) if tau0_true else (tau0_combo["median"] * 1e3),
        savepath=os.path.join(out_dir, "tau0_method_comparison.png"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slopes", default=None, help="slopes.csv (T x 2M)")
    ap.add_argument("--coeffs", default=None, help="zernike_coeffs.csv (T x J)")
    ap.add_argument("--frames", default=None,
                    help="frame dir/glob (compute the time-series; needs --calib)")
    ap.add_argument("--calib", default=None, help="AOMX calib dir (with --frames)")
    ap.add_argument("--config", required=True, help="JSON config path")
    ap.add_argument("--ground-truth", dest="ground_truth", default=None,
                    help="dataset ground_truth.json (else auto-discovered near inputs)")
    ap.add_argument("--dt", type=float, default=None,
                    help="inter-frame interval (s); default = config cadence.dt_s")
    ap.add_argument("--jmax", type=int, default=15)
    ap.add_argument("--out", default="turbulence_summary.json")
    ap.add_argument("--plots", action="store_true")
    # Back-compat single-input form (--in + --kind).
    ap.add_argument("--in", dest="infile", default=None,
                    help="single time-series CSV (use with --kind)")
    ap.add_argument("--kind", choices=["slopes", "zernike", "phase"],
                    default="slopes")
    args = ap.parse_args(argv)

    slopes = args.slopes
    coeffs = args.coeffs
    if args.infile:
        if args.kind == "zernike":
            coeffs = args.infile
        else:
            slopes = args.infile

    analyze(args.config, slopes_path=slopes, coeffs_path=coeffs,
            frames=args.frames, calib_dir=args.calib, out_path=args.out,
            plots=args.plots, dt_s=args.dt, j_max=args.jmax,
            ground_truth_path=args.ground_truth)
    return 0


if __name__ == "__main__":
    sys.exit(main())
