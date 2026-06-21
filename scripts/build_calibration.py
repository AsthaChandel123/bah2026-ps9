#!/usr/bin/env python3
"""build_calibration.py -- config -> AOMX calibration matrices for the C core.

Builds, from a config (and optionally a flat-wavefront reference frame):
  R.aomx          zonal Fried reconstructor      (N x 2M)
  Mpinv.aomx      modal reconstructor M+          (J x 2M)
  Z.aomx          Zernike synthesis Z             (N_pts x J)
  G.aomx          DM command matrix H+*(-1/2)     (N_act x N)
  K.aomx (opt.)   fused slopes->commands G@R      (N_act x 2M)
  refslopes.aomx  reference slopes                (2M x 1)
  subapmap.aomx   sub-aperture windows            (N_sub x 4)

The C real-time core (bin/wfs_rt) mmaps/loads these once at startup.
See ARCHITECTURE.md S2 (offline path), S4.2.

Usage:
  python3 scripts/build_calibration.py --config config/example_config.json --out calib/
"""
from __future__ import annotations

import argparse
import os
import sys


def build_calibration(config_path: str, out_dir: str,
                      flat_frame_path: str | None = None,
                      fuse: bool = False) -> None:
    """Construct all calibration matrices and serialize them as AOMX.

    TODO(impl):
      1. load_config(config_path)
      2. geometry: build_subaperture_grid; register_references(flat) if given;
         active_subaperture_mask; build_actuator_grid; fried_corner_indices
      3. reconstructor: build_fried_gamma -> build_zonal_reconstructor (R);
         build_modal_interaction -> build_modal_reconstructor (M+);
         build_synthesis_matrix (Z)
      4. dm: build_influence_matrix (H) -> build_command_matrix (G);
         optionally fuse_slopes_to_commands (K)
      5. write_aomx each to out_dir; emit a flat key=value config sidecar for
         the C core (so it needs no JSON parser).
    """
    raise NotImplementedError("TODO(impl): build_calibration")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="JSON config path")
    ap.add_argument("--out", default="calib/", help="output directory for AOMX files")
    ap.add_argument("--flat", default=None, help="flat-wavefront reference BMP (optional)")
    ap.add_argument("--fuse", action="store_true", help="also write fused K = G@R")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    build_calibration(args.config, args.out, args.flat, args.fuse)
    return 0


if __name__ == "__main__":
    sys.exit(main())
