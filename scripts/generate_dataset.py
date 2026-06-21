#!/usr/bin/env python3
"""generate_dataset.py -- synthetic SH-WFS .bmp time-series with known r0/tau0.

Generates a frozen-flow phase-screen series at an injected ``r0`` and wind speed
(=> known ``tau0 = 0.314 r0 / v``), synthesizes the spot field (geometric or
Fraunhofer), adds detector noise, and writes zero-padded ``frame_*.bmp`` files
plus a ground-truth JSON (config + injected r0/tau0/wind/L0 + per-frame true
Zernike coefficients). This is the validation backbone (research/07 PART B).

The ground-truth JSON (``<out>/ground_truth.json``) is the validation oracle the
Python pipeline reads back to score r0/tau0 recovery and reconstruction quality.

Usage:
  python3 scripts/generate_dataset.py --config config/example_config.json \
      --r0 0.10 --tau0 0.005 --frames 40 --out results/synthetic/run01 --seed 1234
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aokit.config import load_config        # noqa: E402
from aokit import datagen                    # noqa: E402

GREENWOOD_TAU0_CONST = 0.314                 # tau0 = 0.314 r0 / v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="JSON config path")
    ap.add_argument("--r0", type=float, required=True,
                    help="injected Fried parameter r0 (m)")
    ap.add_argument("--tau0", type=float, default=None,
                    help="injected coherence time tau0 (s); if omitted, derived from --wind")
    ap.add_argument("--wind", type=float, default=None,
                    help="wind speed (m/s); used if --tau0 is omitted")
    ap.add_argument("--wind-angle", type=float, default=0.0,
                    help="wind direction (deg, default 0)")
    ap.add_argument("--L0", type=float, default=25.0,
                    help="outer scale L0 (m); use <=0 or 'inf' for pure Kolmogorov")
    ap.add_argument("--frames", type=int, default=40, help="number of frames")
    ap.add_argument("--model", "--spot-model", dest="model",
                    choices=["geometric", "fraunhofer"], default="geometric",
                    help="spot-field model (geometric tilt oracle or Fraunhofer FFT)")
    ap.add_argument("--jmax", type=int, default=15,
                    help="max Noll index for per-frame ground-truth coefficients")
    ap.add_argument("--flux", type=float, default=5.0e4,
                    help="total photons per spot field (signal strength)")
    ap.add_argument("--read-noise", type=float, default=3.0,
                    help="detector read noise (electrons RMS)")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed")
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    cfg = load_config(args.config)

    # Reconcile tau0 <-> wind. datagen converts tau0 -> v = 0.314 r0 / tau0
    # internally, so pass tau0 when available; otherwise compute it from --wind.
    if args.tau0 is not None and args.tau0 > 0.0:
        tau0 = float(args.tau0)
    elif args.wind is not None and args.wind > 0.0:
        tau0 = GREENWOOD_TAU0_CONST * float(args.r0) / float(args.wind)
    else:
        ap.error("provide either --tau0 or --wind (with a positive value)")
        return 2  # unreachable; ap.error exits

    L0 = float(args.L0)
    L0_m = np.inf if L0 <= 0.0 else L0

    print(f"generate_dataset: r0={args.r0} m, tau0={tau0:.6g} s, "
          f"wind={GREENWOOD_TAU0_CONST * args.r0 / tau0:.4g} m/s, "
          f"L0={'inf' if not np.isfinite(L0_m) else L0_m} m, "
          f"frames={args.frames}, model={args.model}")
    print(f"  frame {cfg.camera.frame_w}x{cfg.camera.frame_h}, "
          f"lenslets {cfg.mla.n_lenslets_x}x{cfg.mla.n_lenslets_y}, "
          f"dt={cfg.dt_s} s -> {args.out}")

    gt = datagen.generate_dataset(
        cfg, r0_m=float(args.r0), tau0_s=tau0, n_frames=int(args.frames),
        out_dir=args.out, spot_model=args.model,
        wind_angle_deg=float(args.wind_angle), L0_m=L0_m,
        flux_photons=float(args.flux), read_noise_e=float(args.read_noise),
        j_max=int(args.jmax), seed=args.seed)

    ff = gt.get("frozen_flow", {})
    print(f"  wrote {gt['n_frames']} frames + ground_truth.json")
    print(f"  injected r0={gt['r0_m']} m  tau0={gt['tau0_s']} s  "
          f"wind={gt['wind_speed_mps']:.4g} m/s  "
          f"shift={ff.get('shift_px_per_frame', float('nan')):.4g} px/frame")
    return 0


if __name__ == "__main__":
    sys.exit(main())
