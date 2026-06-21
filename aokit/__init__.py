"""aokit -- offline toolkit for the ISRO BAH 2026 PS9 SH-WFS pipeline.

This package is the *offline* tier (research/06 S13): it builds the calibration
matrices the C real-time core consumes, generates synthetic SH-WFS datasets
with known injected r0/tau0, runs validation, and serves as the algorithmic
reference oracle.

STABLE CONTRACT: this ``__init__`` defines the public surface that scripts and
tests import. Implementers MUST NOT edit this file -- fill in the submodules
instead. Each submodule currently provides documented signatures that raise
``NotImplementedError`` (algorithmic stages) or are already implemented
(``config``, ``matio``, ``bmpio`` foundations).

Submodules
----------
config        : load/validate JSON config -> Config dataclass
geometry      : Fried sub-aperture & actuator grids, pupil mask, references
zernike       : Noll-indexed Zernike values + analytic gradients
centroiding   : reference centroiders (CoG/TCoG/WCoG/TWCoG/correlation/gaussfit)
reconstructor : zonal Fried LS reconstructor R; modal M+; FTR cross-check
dm            : influence matrix H (Gaussian + coupling), command H+
turbulence    : >=7 r0 estimators + >=6 tau0 estimators + combiner
datagen       : phase screens, spot-field synthesis, noise, frozen-flow series
bmpio         : BMP read/write in pure numpy
matio         : AOMX binary matrix format (byte-matches src/matio.c)
validation    : RMS WFE, Strehl, phase correlation, r0/tau0 recovery, DM residual
viz           : matplotlib plots
"""

__version__ = "0.1.0"

# Submodules are imported lazily-by-name below so that `import aokit` succeeds
# even if a single submodule has an optional-dependency issue. The names are
# the stable contract.
from . import config        # noqa: F401
from . import matio         # noqa: F401
from . import bmpio         # noqa: F401
from . import zernike       # noqa: F401
from . import geometry      # noqa: F401
from . import centroiding   # noqa: F401
from . import reconstructor # noqa: F401
from . import dm            # noqa: F401
from . import turbulence    # noqa: F401
from . import datagen       # noqa: F401
from . import validation    # noqa: F401

# viz imports matplotlib; keep it optional so headless/minimal environments can
# still `import aokit` for the numerical pipeline.
try:
    from . import viz       # noqa: F401
except Exception:  # pragma: no cover - viz is optional
    viz = None

# Convenience re-exports (the most-used entry points).
from .config import Config, load_config  # noqa: F401
from .matio import read_aomx, write_aomx  # noqa: F401

__all__ = [
    "__version__",
    "config", "matio", "bmpio", "zernike", "geometry", "centroiding",
    "reconstructor", "dm", "turbulence", "datagen", "validation", "viz",
    "Config", "load_config", "read_aomx", "write_aomx",
]
