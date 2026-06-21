"""aokit.dm -- DM influence matrix H, command matrix H+, stroke handling.

Builds the Gaussian influence-function matrix ``H`` (with the provided
inter-actuator coupling ``c``), the regularized command matrix
``G = H+ * (-1/2)`` (coupling DECONVOLVED, reflection factor included), and the
stroke-unit conversion. research/05 S1-S4, S7, S9, S11; ARCHITECTURE.md S3.6.

CONVENTIONS (must match the C real-time core, which loads ``G`` as ``G.aomx``)
---------------------------------------------------------------------------
* ``H`` (influence matrix): shape ``(N_phase, N_act)``. Column ``k`` is the DM
  surface produced at every phase node by a *unit* poke of actuator ``k`` alone;
  ``surface = H @ a``. In Fried geometry the phase-reconstruction nodes ARE the
  actuator-corner nodes, so ``N_phase == N_act`` and ``H`` is square.
* ``Hpinv`` (command matrix): shape ``(N_act, N_phase)``. Regularized inverse of
  ``H`` (Tikhonov or truncated SVD). It carries the explicit *anti-coupling*
  (negative) off-diagonal terms that back each actuator off for what its
  neighbours add.
* ``G`` (DM command matrix for the C core): ``G = -0.5 * Hpinv`` -- shape
  ``(N_act, N_phase)``. The factor of ``-1/2`` (phase conjugation ``-W`` plus the
  reflection factor of two: a surface displaced by ``z`` changes the OPD by
  ``2z``) is *baked into* ``G``. The C core computes ``a = G @ phi`` (``phi`` =
  reconstructed wavefront on the nodes), then scales by ``stroke_gain`` and
  clips to ``+/- stroke_max``. Therefore ``G`` already includes the ``-1/2`` --
  do NOT apply it again anywhere else.

INFLUENCE-FUNCTION MODEL
------------------------
The actuator influence function is the power-law (stretched-exponential)
parameterised directly by the coupling ``c`` (research/05 S1.3-1.4)::

    IF(rho) = exp( ln(c) * (rho / pitch) ** alpha )

so a neighbour exactly one pitch away has value ``c`` (the definition of the
inter-actuator coupling coefficient), and a node at distance ``k`` pitches has
value ``c ** (k ** alpha)``. With ``alpha = 2`` this is the axisymmetric
Gaussian ``exp(-rho^2 / (2 sigma^2))`` with width
``sigma = pitch / sqrt(-2 ln c)`` (the KEY relation, research/05 S1.3): at
``rho = pitch`` it drops to ``c``, and a distance-2 node gives ``c**4``,
matching the worked 1-D ``H[i,j] = c**((i-j)**2)`` example (research/05 S9.1).
"""
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from .config import Config
from .geometry import ActuatorGrid


# ----------------------------------------------------------------------------
# Influence-function model
# ----------------------------------------------------------------------------

def gaussian_sigma_from_coupling(pitch_m: float, coupling: float) -> float:
    """Gaussian IF width from coupling: ``sigma = d / sqrt(-2 ln c)`` so the
    bump drops to ``c`` at the actuator pitch ``d``. research/05 S1.3 (KEY
    relation; only valid for the Gaussian ``alpha = 2`` model)."""
    c = float(coupling)
    if not (0.0 < c < 1.0):
        raise ValueError(f"coupling must be in (0, 1); got {c}")
    return float(pitch_m) / np.sqrt(-2.0 * np.log(c))


def influence_function_value(rho: np.ndarray, pitch: float, c: float,
                             alpha: float = 2.0) -> np.ndarray:
    """Evaluate the influence function ``IF(rho)`` at radial distance(s) ``rho``.

    ``IF(rho) = exp( ln(c) * (rho / pitch) ** alpha )`` -- the power-law /
    stretched-exponential model parameterised by the coupling ``c`` (value ``c``
    at one pitch). ``alpha = 2`` is the Gaussian. research/05 S1.3-1.4.
    """
    c = float(c)
    if not (0.0 < c < 1.0):
        raise ValueError(f"coupling c must be in (0, 1); got {c}")
    if pitch <= 0.0:
        raise ValueError("pitch must be positive")
    r = np.abs(np.asarray(rho, dtype=np.float64))
    return np.exp(np.log(c) * (r / float(pitch)) ** float(alpha))


# Backward-compatible alias kept for the stub contract (build_calibration.py /
# other callers may import it). Same model as influence_function_value, with a
# Gaussian fast path when alpha == 2 expressed via sigma.
def influence_function(r: np.ndarray, pitch_m: float, coupling: float,
                       model: str = "gaussian", alpha: float = 2.0) -> np.ndarray:
    """Evaluate IF(r): Gaussian ``exp(-r^2/2 sigma^2)`` (``model='gaussian'``,
    ``alpha=2``) or the power-law ``exp(ln(c) (r/d)^alpha)``. research/05
    S1.3-1.4. Both reduce to value ``coupling`` at one pitch."""
    if model == "gaussian" and float(alpha) == 2.0:
        sigma = gaussian_sigma_from_coupling(pitch_m, coupling)
        rr = np.asarray(r, dtype=np.float64)
        return np.exp(-(rr * rr) / (2.0 * sigma * sigma))
    return influence_function_value(r, pitch_m, coupling, alpha)


# ----------------------------------------------------------------------------
# Influence matrix H
# ----------------------------------------------------------------------------

def _node_positions(geom) -> Tuple[np.ndarray, np.ndarray]:
    """Extract (x, y) actuator-node coordinates from an ActuatorGrid or a
    Geometry (uses ``geom.acts``). Returns pixel coordinates ``(x, y)``."""
    acts = getattr(geom, "acts", geom)
    return np.asarray(acts.x, dtype=np.float64), np.asarray(acts.y, dtype=np.float64)


def gaussian_influence_matrix(geom, coupling_c: float, alpha: float = 2.0
                              ) -> np.ndarray:
    """Influence matrix ``H`` (``N_phase`` x ``N_act``) from the Gaussian /
    power-law IF model. research/05 S1-S2.

    Column ``k`` is actuator ``k``'s influence function sampled at every phase
    node: ``H[i, k] = IF(|node_i - node_k|)`` with
    ``IF(rho) = exp(ln(c) * (rho / pitch) ** alpha)`` (value ``coupling_c`` at one
    pitch; ``alpha = 2`` is the Gaussian). In Fried geometry the phase nodes are
    the actuator-corner nodes, so the result is the square ``(N_act, N_act)``
    coupling matrix with unit diagonal, ``coupling_c`` on the nearest-neighbour
    off-diagonals, and ``coupling_c ** (k**alpha)`` at distance ``k`` pitches.

    Parameters
    ----------
    geom : ActuatorGrid | Geometry
        Provides actuator node coordinates (``geom.acts`` if a Geometry).
    coupling_c : float
        Inter-actuator coupling in ``(0, 1)`` -- the IF value at one pitch.
    alpha : float
        Power index of the IF skirt (``2.0`` => Gaussian).

    Returns
    -------
    H : np.ndarray, shape (N_phase, N_act)
        Surface-from-commands matrix; ``surface = H @ a``.
    """
    c = float(coupling_c)
    if not (0.0 < c < 1.0):
        raise ValueError(f"coupling_c must be in (0, 1); got {c}")

    x, y = _node_positions(geom)
    n = x.shape[0]

    # Pitch in the same (pixel) units as the node coordinates. ActuatorGrid does
    # not carry pitch, so infer it from the (n_x+1)x(n_y+1) row-major corner grid
    # via the (col,row) index array and the node coordinates: the spacing between
    # adjacent columns of row 0 is one pitch.
    pitch = _infer_pitch(geom, x, y)

    # Pairwise distances between all node pairs.
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    rho = np.sqrt(dx * dx + dy * dy)

    H = influence_function_value(rho, pitch, c, alpha)
    return np.ascontiguousarray(H, dtype=np.float64)


def _infer_pitch(geom, x: np.ndarray, y: np.ndarray) -> float:
    """Infer the actuator pitch (in node-coordinate units) from the grid.

    Prefers the row-major ``(col, row)`` index layout of an ``ActuatorGrid`` to
    find two horizontally adjacent nodes; falls back to the smallest positive
    pairwise distance.
    """
    acts = getattr(geom, "acts", geom)
    ij = getattr(acts, "ij", None)
    if ij is not None:
        ij = np.asarray(ij)
        # nodes with row == row.min(): find a pair with adjacent columns.
        row0 = ij[:, 1] == ij[:, 1].min()
        cols = ij[row0, 0]
        idx = np.where(row0)[0]
        order = np.argsort(cols)
        idx = idx[order]
        cols = cols[order]
        for a in range(len(idx) - 1):
            if cols[a + 1] - cols[a] == 1:
                i0, i1 = idx[a], idx[a + 1]
                d = float(np.hypot(x[i1] - x[i0], y[i1] - y[i0]))
                if d > 0.0:
                    return d
    # Fallback: smallest non-zero pairwise distance.
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    r = np.sqrt(dx * dx + dy * dy)
    r = r[r > 1e-12]
    if r.size == 0:
        raise ValueError("cannot infer actuator pitch: nodes coincide")
    return float(r.min())


def build_influence_matrix(cfg: Config, acts: ActuatorGrid,
                           sample_xy: np.ndarray) -> np.ndarray:
    """Influence matrix ``H`` (``N_pts`` x ``N_act``): column ``a`` is actuator
    ``a``'s IF sampled at ``sample_xy``. Surface = ``H @ a``. research/05 S2.

    ``sample_xy`` is an ``(N_pts, 2)`` array of ``(x, y)`` sample coordinates (in
    the same pixel units as ``acts.x/acts.y``); e.g. the phase grid. Uses the
    config's coupling ``c``, influence model and ``alpha``. This is the general
    (possibly non-square) sampling used by the calibration builder; for the Fried
    node-on-node case use :func:`gaussian_influence_matrix`.
    """
    ax = np.asarray(acts.x, dtype=np.float64)
    ay = np.asarray(acts.y, dtype=np.float64)
    pts = np.asarray(sample_xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("sample_xy must be (N_pts, 2)")

    c = float(cfg.dm.coupling_coeff)
    alpha = float(cfg.dm.influence_alpha)
    pitch = float(cfg.px_per_lenslet)  # DM pitch in detector px (Fried: == lenslet)

    dx = pts[:, 0][:, None] - ax[None, :]
    dy = pts[:, 1][:, None] - ay[None, :]
    rho = np.sqrt(dx * dx + dy * dy)
    H = influence_function_value(rho, pitch, c, alpha)
    return np.ascontiguousarray(H, dtype=np.float64)


# ----------------------------------------------------------------------------
# Command matrix Hpinv and DM command matrix G
# ----------------------------------------------------------------------------

def command_matrix(H: np.ndarray, reg: str = "tikhonov", mu: float = 1e-3,
                   sv_threshold: float = 1e-6) -> np.ndarray:
    """Regularized command matrix ``Hpinv`` (``N_act`` x ``N_phase``) inverting
    the influence matrix ``H`` (``N_phase`` x ``N_act``). research/05 S3-S4, S9.

    Two regularizations (both damp the small singular values that would amplify
    noise / waffle into huge commands, research/05 S4.1-4.2):

    * ``reg='tikhonov'`` (damped least squares):
      ``Hpinv = (H^T H + mu^2 I)^-1 H^T``. In SVD form this is
      ``sum_k [sigma_k / (sigma_k^2 + mu^2)] v_k u_k^T``. ``mu`` is the damping.
    * ``reg='tsvd'`` (truncated SVD): drop modes with
      ``sigma_k < sv_threshold * sigma_max`` (set their inverse to zero).

    For a well-conditioned coupling matrix (typical small ``c``) and small
    ``mu``, the result is very close to the exact inverse and carries the
    negative anti-coupling off-diagonals (research/05 S9.1). Conditioning is
    handled entirely here (offline, once); the per-frame path is a pure MVM.
    """
    H = np.asarray(H, dtype=np.float64)
    if H.ndim != 2:
        raise ValueError("H must be 2-D")

    if reg == "tikhonov":
        # SVD form is numerically stable and works for non-square H.
        U, s, Vt = np.linalg.svd(H, full_matrices=False)
        d = s / (s * s + float(mu) ** 2)
        # Hpinv = V diag(d) U^T  (shape N_act x N_phase)
        Hpinv = (Vt.T * d) @ U.T
        return np.ascontiguousarray(Hpinv, dtype=np.float64)

    if reg in ("tsvd", "truncated", "truncated_svd"):
        U, s, Vt = np.linalg.svd(H, full_matrices=False)
        smax = s.max() if s.size else 0.0
        keep = s > float(sv_threshold) * smax
        d = np.zeros_like(s)
        d[keep] = 1.0 / s[keep]
        Hpinv = (Vt.T * d) @ U.T
        return np.ascontiguousarray(Hpinv, dtype=np.float64)

    raise ValueError(f"unknown reg '{reg}' (use 'tikhonov' or 'tsvd')")


def dm_command_matrix(Hpinv: np.ndarray) -> np.ndarray:
    """Fuse the reflection / conjugation factor into the DM command matrix:
    ``G = -0.5 * Hpinv`` (shape ``N_act`` x ``N_phase``). research/05 S3.1.

    The factor ``-1/2`` = phase conjugation (``-W``) times the reflection factor
    of two (surface displaced ``z`` -> OPD ``2z``). The C core then computes
    ``a = G @ phi``; ``G`` already contains the ``-1/2`` so it is not applied
    again elsewhere.
    """
    return np.ascontiguousarray(-0.5 * np.asarray(Hpinv, dtype=np.float64))


def build_command_matrix(H: np.ndarray, method: str = "tikhonov",
                         tikhonov_mu: float = 1e-3,
                         sv_threshold: float = 1e-6,
                         reflection_factor: float = -0.5,
                         remove_waffle: bool = True) -> np.ndarray:
    """Regularized DM command matrix ``G = reflection_factor * Hpinv`` (so
    ``a = G @ phi = Hpinv @ (-phi/2)``). research/05 S3, S4, S9.

    Convenience wrapper that builds ``Hpinv`` via :func:`command_matrix` (method
    ``'tikhonov'`` or ``'tsvd'``/``'svd'``) and bakes in the reflection factor.
    Coupling is encoded in ``H``, so ``G`` carries the anti-coupling
    off-diagonals. ``remove_waffle`` is accepted for API compatibility; the
    truncated-SVD / Tikhonov damping already suppresses the low-singular-value
    (waffle-like) modes that the Fried geometry is blind to.
    """
    reg = "tikhonov" if method == "tikhonov" else "tsvd"
    Hpinv = command_matrix(H, reg=reg, mu=tikhonov_mu, sv_threshold=sv_threshold)
    return np.ascontiguousarray(float(reflection_factor) * Hpinv,
                                dtype=np.float64)


# ----------------------------------------------------------------------------
# Per-frame command computation + strokes
# ----------------------------------------------------------------------------

def actuator_commands(W_nodes: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Actuator commands ``a = G @ W_nodes`` (units of actuator stroke, before
    the ``stroke_gain``). research/05 S3; ARCHITECTURE.md S2.

    ``W_nodes`` is the reconstructed wavefront sampled on the phase/actuator
    nodes (length ``N_phase``); ``G`` already contains the ``-1/2`` reflection
    factor and the deconvolved coupling, so ``a`` is the conjugate-target stroke
    map ``Hpinv @ (-W_nodes / 2)``. This is the single per-frame matrix-vector
    multiply.
    """
    G = np.asarray(G, dtype=np.float64)
    W = np.asarray(W_nodes, dtype=np.float64)
    return G @ W


def fuse_slopes_to_commands(G: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Fused slopes->commands matrix ``K = G @ R`` (single MVM path). The C core
    can run ``a = K @ s`` directly when the wavefront map is not also needed.
    research/02 S12; ARCHITECTURE.md S2."""
    return np.ascontiguousarray(
        np.asarray(G, dtype=np.float64) @ np.asarray(R, dtype=np.float64))


def apply_stroke_clip(a: np.ndarray, stroke_max: float) -> Tuple[np.ndarray, int]:
    """Clip commands to ``[-stroke_max, +stroke_max]``; return
    ``(clipped, n_saturated)``. research/05 S4.3.

    ``n_saturated`` counts the entries whose magnitude *exceeded* ``stroke_max``
    (and were therefore clipped). Physical actuators have a finite stroke; values
    beyond the envelope are saturated.
    """
    a = np.asarray(a, dtype=np.float64)
    amax = float(stroke_max)
    n_sat = int(np.count_nonzero(np.abs(a) > amax))
    return np.clip(a, -amax, amax), n_sat


def clip_strokes(commands: np.ndarray, stroke_max: float) -> Tuple[np.ndarray, int]:
    """Alias of :func:`apply_stroke_clip` kept for the stub contract: clip to
    ``+/- stroke_max`` and return ``(clipped, n_saturated)``. research/05 S4.3."""
    return apply_stroke_clip(commands, stroke_max)


def to_stroke_units(commands: np.ndarray, gain_m_per_unit: float) -> np.ndarray:
    """Convert unit commands to physical stroke length (m) via the gain ``g``:
    ``z[m] = g * a``. research/05 S7. (The ``-1/2`` reflection factor lives in
    ``G``/the command path, not here.)"""
    return np.asarray(commands, dtype=np.float64) * float(gain_m_per_unit)


def build_interaction_matrix(poke_slopes: np.ndarray, poke_amp: float
                             ) -> np.ndarray:
    """Calibration-based interaction matrix ``D = d slopes / d command`` from
    poke frames: ``D[:, j] = poke_slopes[:, j] / poke_amp`` (alternative to the
    model ``H``). research/05 S6."""
    return np.ascontiguousarray(
        np.asarray(poke_slopes, dtype=np.float64) / float(poke_amp))


# ----------------------------------------------------------------------------
# Convenience builder for the calibration step (Wave 3 writes AOMX)
# ----------------------------------------------------------------------------

def build_dm(geom, cfg: Config, reg: str = "tikhonov",
             mu: Optional[float] = None) -> dict:
    """Convenience builder returning the DM matrices for the calibration step.

    Builds the Fried node-on-node influence matrix ``H`` from the config coupling
    and influence model, the regularized command matrix ``Hpinv``, and the fused
    DM command matrix ``G = -0.5 * Hpinv``. research/05 S2-S4; ARCHITECTURE.md
    S3.6. Returns only arrays; serialization to AOMX is done by the calibration
    builder (Wave 3).

    Returns
    -------
    dict with keys ``'H'`` (N_phase x N_act), ``'Hpinv'`` (N_act x N_phase) and
    ``'G'`` (N_act x N_phase = -0.5 * Hpinv).
    """
    c = float(cfg.dm.coupling_coeff)
    alpha = float(cfg.dm.influence_alpha)
    if mu is None:
        mu = 1e-3
    H = gaussian_influence_matrix(geom, c, alpha=alpha)
    Hpinv = command_matrix(H, reg=reg, mu=mu)
    G = dm_command_matrix(Hpinv)
    return {"H": H, "Hpinv": Hpinv, "G": G}
