"""aokit.config -- load/validate the JSON config into a Config dataclass.

Implemented (foundational). Mirrors the schema in ARCHITECTURE.md S4.1 and the
C struct in ``src/aoconfig.h``. The ``ground_truth`` block is optional (present
only for synthetic datasets produced by scripts/generate_dataset.py).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class CameraCfg:
    pixel_size_m: float
    frame_w: int
    frame_h: int
    bit_depth: int = 8


@dataclass
class MlaCfg:
    n_lenslets_x: int
    n_lenslets_y: int
    pitch_m: float
    focal_length_m: float


@dataclass
class PupilCfg:
    diameter_m: float
    center_x_px: float
    center_y_px: float


@dataclass
class DmCfg:
    n_act_x: int
    n_act_y: int
    pitch_m: float
    coupling_coeff: float
    stroke_max_m: float
    influence_model: str = "gaussian"
    influence_alpha: float = 2.0
    stroke_gain_m_per_unit: float = 1.0e-6


@dataclass
class GeometryCfg:
    type: str = "fried"
    rotation_deg: float = 0.0
    flip_y: bool = False


@dataclass
class GroundTruth:
    r0_m: Optional[float] = None
    tau0_s: Optional[float] = None
    wind_speed_mps: Optional[float] = None
    L0_m: Optional[float] = None
    zernike_noll: List[float] = field(default_factory=list)


@dataclass
class Config:
    schema_version: int
    camera: CameraCfg
    mla: MlaCfg
    pupil: PupilCfg
    wavelength_m: float
    dm: DmCfg
    geometry: GeometryCfg
    dt_s: float
    ground_truth: Optional[GroundTruth] = None

    # ---- derived convenience properties ----
    @property
    def n_sub_nominal(self) -> int:
        return self.mla.n_lenslets_x * self.mla.n_lenslets_y

    @property
    def n_act_nominal(self) -> int:
        return self.dm.n_act_x * self.dm.n_act_y

    @property
    def px_per_lenslet(self) -> float:
        """Detector pixels per lenslet pitch (needs pitch & pixel_size in
        the same plane; here both are referenced to the pupil/detector grid)."""
        return self.mla.pitch_m / self.camera.pixel_size_m

    @property
    def slope_scale(self) -> float:
        """Multiply (centroid - ref) in px by this to get slope in rad."""
        return self.camera.pixel_size_m / self.mla.focal_length_m


def load_config(path: str) -> Config:
    """Load and validate a JSON config file into a :class:`Config`."""
    with open(path, "r") as f:
        d = json.load(f)
    return from_dict(d)


def from_dict(d: dict) -> Config:
    """Build a :class:`Config` from a parsed dict (validating required keys)."""
    cam = d["camera"]
    mla = d["mla"]
    pup = d["pupil"]
    dm = d["dm"]
    geo = d.get("geometry", {})
    cad = d.get("cadence", {})

    cfg = Config(
        schema_version=int(d.get("schema_version", 1)),
        camera=CameraCfg(
            pixel_size_m=float(cam["pixel_size_m"]),
            frame_w=int(cam["frame_w"]),
            frame_h=int(cam["frame_h"]),
            bit_depth=int(cam.get("bit_depth", 8)),
        ),
        mla=MlaCfg(
            n_lenslets_x=int(mla["n_lenslets_x"]),
            n_lenslets_y=int(mla["n_lenslets_y"]),
            pitch_m=float(mla["pitch_m"]),
            focal_length_m=float(mla["focal_length_m"]),
        ),
        pupil=PupilCfg(
            diameter_m=float(pup["diameter_m"]),
            center_x_px=float(pup["center_x_px"]),
            center_y_px=float(pup["center_y_px"]),
        ),
        wavelength_m=float(d["wavelength_m"]),
        dm=DmCfg(
            n_act_x=int(dm["n_act_x"]),
            n_act_y=int(dm["n_act_y"]),
            pitch_m=float(dm["pitch_m"]),
            coupling_coeff=float(dm["coupling_coeff"]),
            stroke_max_m=float(dm["stroke_max_m"]),
            influence_model=str(dm.get("influence_model", "gaussian")),
            influence_alpha=float(dm.get("influence_alpha", 2.0)),
            stroke_gain_m_per_unit=float(dm.get("stroke_gain_m_per_unit", 1.0e-6)),
        ),
        geometry=GeometryCfg(
            type=str(geo.get("type", "fried")),
            rotation_deg=float(geo.get("rotation_deg", 0.0)),
            flip_y=bool(geo.get("flip_y", False)),
        ),
        dt_s=float(cad.get("dt_s", d.get("dt_s", 2.0e-3))),
    )

    gt = d.get("ground_truth")
    if gt is not None:
        cfg.ground_truth = GroundTruth(
            r0_m=gt.get("r0_m"),
            tau0_s=gt.get("tau0_s"),
            wind_speed_mps=gt.get("wind_speed_mps"),
            L0_m=gt.get("L0_m"),
            zernike_noll=list(gt.get("zernike_noll", [])),
        )

    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    """Sanity-check a config; raise ValueError on inconsistency.

    Notably warns (via exception) if the geometry is Fried but the actuator
    grid is not (n_lenslets + 1) in each axis, since PS9 mandates Fried.
    """
    if cfg.camera.bit_depth not in (8, 24):
        raise ValueError("camera.bit_depth must be 8 or 24")
    if cfg.mla.focal_length_m <= 0:
        raise ValueError("mla.focal_length_m must be positive")
    if not (0.0 <= cfg.dm.coupling_coeff < 1.0):
        raise ValueError("dm.coupling_coeff must be in [0, 1)")
    if cfg.geometry.type == "fried":
        if (cfg.dm.n_act_x != cfg.mla.n_lenslets_x + 1 or
                cfg.dm.n_act_y != cfg.mla.n_lenslets_y + 1):
            raise ValueError(
                "Fried geometry expects n_act == n_lenslets + 1 in each axis "
                f"(got act {cfg.dm.n_act_x}x{cfg.dm.n_act_y}, "
                f"lenslets {cfg.mla.n_lenslets_x}x{cfg.mla.n_lenslets_y})"
            )
