#!/usr/bin/env python3
"""analyze_turbulence.py -- multi-method r0/tau0 report from a time-series.

Reads a slopes / Zernike-coefficient / phase-map time-series (as written by the
C core or run_pipeline_py.py), runs >=7 r0 estimators and >=6 tau0 estimators,
removes biases, reconciles them (median +/- spread), and writes
``turbulence_summary.json`` (+ optional plots). research/04 S5.

Usage:
  python3 scripts/analyze_turbulence.py --in out/slopes.csv \
      --config config/example_config.json --out out/turbulence_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="infile", required=True,
                    help="slopes.csv / zernike_coeffs.csv / phase series (AOMX)")
    ap.add_argument("--kind", choices=["slopes", "zernike", "phase"],
                    default="slopes", help="type of the input time-series")
    ap.add_argument("--config", required=True, help="JSON config path")
    ap.add_argument("--out", default="turbulence_summary.json")
    ap.add_argument("--plots", action="store_true")
    args = ap.parse_args(argv)

    # TODO(impl):
    #   cfg = load_config(args.config)
    #   load the time-series (slopes_ts / coeffs_ts / phase_ts)
    #   res = turbulence.estimate_all(slopes_ts, coeffs_ts, phase_ts, cfg, cfg.dt_s)
    #   json.dump(res.to_dict(), open(args.out, 'w'), indent=2)
    #   if --plots: viz.plot_turbulence_trends(...)
    raise NotImplementedError("TODO(impl): analyze_turbulence entrypoint")


if __name__ == "__main__":
    sys.exit(main())
