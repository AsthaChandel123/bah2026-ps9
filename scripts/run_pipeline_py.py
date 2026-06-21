#!/usr/bin/env python3
"""run_pipeline_py.py -- pure-Python end-to-end pipeline (validation/plots).

Mirrors the C real-time pipeline in Python for validation and visualization:
  BMP -> centroid -> slopes -> {zonal R, modal M+} -> phase W, Zernike a ->
  DM actuator map (G). Produces per-frame outputs and (optionally) plots, and
  is the oracle the C core is validated against (C/Python parity).

Usage:
  python3 scripts/run_pipeline_py.py --config config/example_config.json \
      --frames "data/synthetic/run01/frame_*.bmp" --out out_py/
"""
from __future__ import annotations

import argparse
import glob
import os
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="JSON config path")
    ap.add_argument("--frames", required=True,
                    help="glob (e.g. 'dir/frame_*.bmp') or a directory")
    ap.add_argument("--calib", default=None,
                    help="optional AOMX calib dir; if omitted, matrices are built in-memory")
    ap.add_argument("--out", default="out_py/", help="output directory")
    ap.add_argument("--plots", action="store_true", help="emit matplotlib plots")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    frame_paths = sorted(glob.glob(args.frames)) if not os.path.isdir(args.frames) \
        else sorted(glob.glob(os.path.join(args.frames, "*.bmp")))

    # TODO(impl):
    #   cfg = load_config(args.config)
    #   build/load R, M+, Z, G (reconstructor.*, dm.*)
    #   for each frame: bmpio.read_bmp_gray -> centroiding.twcog per sub-ap ->
    #       slopes -> phi = R@s ; a = M+@s ; W = Z@a ; acts = G@phi (clip+gain)
    #   write phase maps / zernike_coeffs.csv / actuator maps / slopes.csv
    #   if --plots: viz.* (spot field, phase, spectrum, actuator map)
    raise NotImplementedError("TODO(impl): run_pipeline_py entrypoint")


if __name__ == "__main__":
    sys.exit(main())
