#!/usr/bin/env python3
"""generate_dataset.py -- synthetic SH-WFS .bmp time-series with known r0/tau0.

Generates a frozen-flow phase-screen series at an injected ``r0`` and wind speed
(=> known ``tau0 = 0.314 r0 / v``), synthesizes the spot field (geometric or
Fraunhofer), adds detector noise, and writes zero-padded ``frame_*.bmp`` files
plus a ground-truth JSON (config + injected r0/tau0/wind/L0). This is the
validation backbone (research/07 PART B).

Usage:
  python3 scripts/generate_dataset.py --config config/example_config.json \
      --r0 0.15 --tau0 0.0045 --frames 2000 --out data/synthetic/run01
"""
from __future__ import annotations

import argparse
import os
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="JSON config path")
    ap.add_argument("--r0", type=float, required=True, help="injected Fried parameter (m)")
    ap.add_argument("--tau0", type=float, default=None,
                    help="injected coherence time (s); if omitted, derived from --wind")
    ap.add_argument("--wind", type=float, default=None, help="wind speed (m/s)")
    ap.add_argument("--wind-angle", type=float, default=0.0, help="wind direction (deg)")
    ap.add_argument("--L0", type=float, default=25.0, help="outer scale (m)")
    ap.add_argument("--frames", type=int, default=1000, help="number of frames")
    ap.add_argument("--spot-model", choices=["geometric", "fraunhofer"],
                    default="geometric")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)

    # TODO(impl):
    #   cfg = load_config(args.config)
    #   tau0/wind reconciliation: tau0 = 0.314 * r0 / wind
    #   gt = datagen.generate_dataset(cfg, r0, tau0, frames, out, spot_model, seed)
    #   write <out>/meta.json with cfg + ground_truth (gt)
    raise NotImplementedError("TODO(impl): generate_dataset entrypoint")


if __name__ == "__main__":
    sys.exit(main())
