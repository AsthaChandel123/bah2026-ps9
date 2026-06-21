"""Unit tests for aokit.dm (influence matrix H, command matrix G, strokes).

Central property (research/05): inter-actuator coupling is DECONVOLVED, not
naively sampled. Tests reproduce the research/05 S9 worked 3-actuator example
(c=0.15 and c=0.35), check the anti-coupling off-diagonal signs, the
reconstruction round-trip (H @ (G @ W) == -W/2), stroke clipping, and the
shape/factor conventions consumed by the C core.
"""
import numpy as np
import pytest

from aokit import dm
from aokit.config import from_dict
from aokit.geometry import build_actuator_grid, build_geometry


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

class _LineActs:
    """Minimal ActuatorGrid-like object: N actuators on a line, pitch 1.

    Provides the ``.x``, ``.y`` and ``.ij`` (col,row) attributes that
    ``gaussian_influence_matrix`` reads, matching the research/05 S9.1 1-D layout.
    """

    def __init__(self, n=3):
        self.x = np.arange(n, dtype=np.float64)
        self.y = np.zeros(n, dtype=np.float64)
        self.ij = np.stack([np.arange(n), np.zeros(n, dtype=int)], axis=1)


def _example_config():
    """A small Fried config: 4x4 lenslets -> 5x5 = 25 actuator nodes."""
    return from_dict({
        "schema_version": 1,
        "camera": {"pixel_size_m": 5.5e-6, "frame_w": 256, "frame_h": 256,
                   "bit_depth": 8},
        "mla": {"n_lenslets_x": 4, "n_lenslets_y": 4, "pitch_m": 1.5e-4,
                "focal_length_m": 5.2e-3},
        "pupil": {"diameter_m": 1.2e-3, "center_x_px": 128.0,
                  "center_y_px": 128.0},
        "wavelength_m": 6.33e-7,
        "dm": {"n_act_x": 5, "n_act_y": 5, "pitch_m": 1.5e-4,
               "coupling_coeff": 0.15, "stroke_max_m": 3.5e-6,
               "influence_model": "gaussian", "influence_alpha": 2.0,
               "stroke_gain_m_per_unit": 1.0e-6},
        "geometry": {"type": "fried", "rotation_deg": 0.0, "flip_y": False},
        "cadence": {"dt_s": 2.0e-3},
    })


# ----------------------------------------------------------------------------
# Influence-function model / width
# ----------------------------------------------------------------------------

def test_gaussian_sigma_from_coupling():
    """sigma = d/sqrt(-2 ln c): c=0.15 -> sigma~0.513 d; c=0.135 -> sigma~d/2
    (research/05 S1.3)."""
    assert dm.gaussian_sigma_from_coupling(1.0, 0.15) == pytest.approx(0.513, abs=2e-3)
    assert dm.gaussian_sigma_from_coupling(1.0, np.exp(-2.0)) == pytest.approx(0.5, abs=1e-9)
    # scales with pitch
    assert dm.gaussian_sigma_from_coupling(3.0, 0.15) == pytest.approx(3 * 0.513, abs=6e-3)


def test_influence_function_value_at_pitch():
    """IF(rho)=exp(ln(c)(rho/d)^alpha): value 1 at center, c at one pitch,
    c**(k**alpha) at k pitches (research/05 S1.3-1.4)."""
    c = 0.15
    assert dm.influence_function_value(0.0, 1.0, c, 2.0) == pytest.approx(1.0)
    assert dm.influence_function_value(1.0, 1.0, c, 2.0) == pytest.approx(c)
    assert dm.influence_function_value(2.0, 1.0, c, 2.0) == pytest.approx(c ** 4)
    # alpha is the power index; alpha=2 is the Gaussian form
    sigma = dm.gaussian_sigma_from_coupling(1.0, c)
    rho = np.array([0.0, 0.5, 1.0, 1.5])
    gauss = np.exp(-(rho ** 2) / (2 * sigma ** 2))
    assert np.allclose(dm.influence_function_value(rho, 1.0, c, 2.0), gauss)


def test_influence_value_rejects_bad_coupling():
    with pytest.raises(ValueError):
        dm.influence_function_value(1.0, 1.0, 0.0, 2.0)
    with pytest.raises(ValueError):
        dm.influence_function_value(1.0, 1.0, 1.0, 2.0)


# ----------------------------------------------------------------------------
# Influence matrix H structure
# ----------------------------------------------------------------------------

def test_worked_example_H_matrix():
    """research/05 S9.1: the 1-D 3-actuator c=0.15 coupling matrix has unit
    diagonal, 0.15 nearest-neighbour, ~0.0005 next-neighbour, det~0.955."""
    acts = _LineActs(3)
    H = dm.gaussian_influence_matrix(acts, 0.15, alpha=2.0)
    assert H.shape == (3, 3)
    expected = np.array([
        [1.000, 0.150, 0.15 ** 4],
        [0.150, 1.000, 0.150],
        [0.15 ** 4, 0.150, 1.000],
    ])
    assert np.allclose(H, expected, atol=1e-6)
    assert np.allclose(H, H.T)                       # symmetric
    assert float(np.linalg.det(H)) == pytest.approx(0.955, abs=2e-3)


def test_influence_matrix_is_banded():
    """H is sparse/banded: each surface point sees only nearby actuators -- the
    next-neighbour term is c**4 << c, far ones ~0 (research/05 S2)."""
    acts = _LineActs(6)
    c = 0.15
    H = dm.gaussian_influence_matrix(acts, c, alpha=2.0)
    assert np.allclose(np.diag(H), 1.0)
    # nearest neighbour == c, distance-2 == c**4, distance-3 == c**9 (tiny)
    assert H[0, 1] == pytest.approx(c)
    assert H[0, 2] == pytest.approx(c ** 4)
    assert H[0, 3] == pytest.approx(c ** 9)
    assert H[0, 3] < 1e-6                              # effectively banded


def test_influence_matrix_square_for_fried_geom():
    """In Fried geometry H is square (N_act, N_act) since phase nodes == actuator
    nodes (research/05 S5; ARCHITECTURE.md S3.6)."""
    cfg = _example_config()
    geom = build_geometry(cfg)
    H = dm.gaussian_influence_matrix(geom, cfg.dm.coupling_coeff,
                                     alpha=cfg.dm.influence_alpha)
    n_act = geom.acts.x.shape[0]
    assert H.shape == (n_act, n_act)
    assert n_act == cfg.dm.n_act_x * cfg.dm.n_act_y       # 25
    assert np.allclose(np.diag(H), 1.0)
    assert np.allclose(H, H.T)


# ----------------------------------------------------------------------------
# Command matrix Hpinv / G -- anti-coupling and worked numbers
# ----------------------------------------------------------------------------

def test_command_matrix_has_anti_coupling_offdiagonals():
    """Hpinv (and G = -1/2 Hpinv) contain negative off-diagonal 'anti-coupling'
    terms (research/05 S9.1): for the 3-actuator c=0.15 example the inverse is
    [[1.0235,-0.157,0.023],...] with the documented -0.157 off-diagonal."""
    acts = _LineActs(3)
    H = dm.gaussian_influence_matrix(acts, 0.15, alpha=2.0)
    Hpinv = dm.command_matrix(H, reg="tikhonov", mu=1e-6)
    expected_inv = np.array([
        [1.0235, -0.1570, 0.0230],
        [-0.1570, 1.0471, -0.1570],
        [0.0230, -0.1570, 1.0235],
    ])
    assert np.allclose(Hpinv, expected_inv, atol=2e-3)
    # nearest-neighbour off-diagonals are negative (anti-coupling)
    assert Hpinv[0, 1] < 0.0 and Hpinv[1, 2] < 0.0
    # the fused G keeps the same (negated) sign structure
    G = dm.dm_command_matrix(Hpinv)
    assert G[0, 1] > 0.0      # -0.5 * (negative) -> positive, but still off-diag


def test_worked_example_deconvolved_vs_naive():
    """research/05 S9.1: naive sampling a=s_target overshoots (H@a != s_target);
    the deconvolved a=Hpinv@s_target = [0.366,0.890,0.366] reproduces s_target."""
    acts = _LineActs(3)
    H = dm.gaussian_influence_matrix(acts, 0.15, alpha=2.0)
    Hpinv = dm.command_matrix(H, reg="tikhonov", mu=1e-6)
    s_target = np.array([0.5, 1.0, 0.5])

    # Naive: send the conjugate samples directly -> overshoot to [0.65,1.15,0.65]
    naive = s_target.copy()
    surf_naive = H @ naive
    assert np.allclose(surf_naive, [0.650, 1.150, 0.650], atol=2e-3)
    assert np.all(surf_naive > s_target)                       # over-corrected

    # Deconvolved: commands are SMALLER and reproduce the target exactly
    a = Hpinv @ s_target
    assert np.allclose(a, [0.366, 0.890, 0.366], atol=2e-3)
    assert np.all(a < naive)                                   # backed off
    assert np.allclose(H @ a, s_target, atol=1e-6)


def test_naive_sampling_overshoots():
    """Naive a = s_target (no deconvolution) makes H@a OVERSHOOT the target,
    demonstrating why coupling must be inverted (research/05 S3.2, S9.1)."""
    acts = _LineActs(5)
    H = dm.gaussian_influence_matrix(acts, 0.2, alpha=2.0)
    s_target = np.array([0.3, 0.6, 1.0, 0.6, 0.3])
    surf_naive = H @ s_target
    # interior points (which have two neighbours) overshoot
    assert surf_naive[2] > s_target[2]
    assert surf_naive[1] > s_target[1]
    # the deconvolved solve removes the overshoot
    Hpinv = dm.command_matrix(H, reg="tikhonov", mu=1e-8)
    assert np.allclose(H @ (Hpinv @ s_target), s_target, atol=1e-6)


def test_larger_coupling_backs_off_more():
    """research/05 S9.3: with c=0.35 the corrected commands shrink further than
    with c=0.15 (naive 0.5/1.0/0.5 -> [0.195,0.864,0.195]) and the anti-coupling
    off-diagonal grows to ~ -0.4545."""
    acts = _LineActs(3)
    s_target = np.array([0.5, 1.0, 0.5])

    H15 = dm.gaussian_influence_matrix(acts, 0.15, alpha=2.0)
    H35 = dm.gaussian_influence_matrix(acts, 0.35, alpha=2.0)
    a15 = dm.command_matrix(H15, reg="tikhonov", mu=1e-9) @ s_target
    Hpinv35 = dm.command_matrix(H35, reg="tikhonov", mu=1e-9)
    a35 = Hpinv35 @ s_target

    assert np.allclose(a35, [0.195, 0.864, 0.195], atol=3e-3)
    # larger coupling -> commands backed off further (smaller)
    assert np.all(a35 < a15)
    # and stronger anti-coupling off-diagonal
    assert Hpinv35[0, 1] == pytest.approx(-0.4545, abs=3e-3)
    assert abs(Hpinv35[0, 1]) > abs(dm.command_matrix(H15, mu=1e-9)[0, 1])


# ----------------------------------------------------------------------------
# Reconstruction consistency on a 2-D Fried grid
# ----------------------------------------------------------------------------

def test_reconstruction_roundtrip_realizes_minus_half_W():
    """For a representable desired wavefront W on the nodes, commands a = G@W give
    a realized surface H@a ~ -W/2 within tolerance (ARCHITECTURE.md S2; G already
    carries the -1/2 reflection factor)."""
    cfg = _example_config()
    geom = build_geometry(cfg)
    built = dm.build_dm(geom, cfg, reg="tikhonov", mu=1e-6)
    H, G = built["H"], built["G"]
    n = H.shape[0]

    rng = np.random.default_rng(0)
    W = rng.standard_normal(n)
    a = dm.actuator_commands(W, G)            # a = G @ W = Hpinv @ (-W/2)
    realized = H @ a                           # realized DM surface
    assert np.allclose(realized, -0.5 * W, atol=1e-6)


def test_G_equals_minus_half_Hpinv():
    """G == -0.5 * Hpinv (fused reflection) -- the convention the C core relies
    on (ARCHITECTURE.md S4.2 G.aomx)."""
    cfg = _example_config()
    geom = build_geometry(cfg)
    built = dm.build_dm(geom, cfg)
    assert np.allclose(built["G"], -0.5 * built["Hpinv"])
    assert built["H"].shape[0] == built["H"].shape[1]          # square (Fried)
    n_act = geom.acts.x.shape[0]
    assert built["Hpinv"].shape == (n_act, n_act)
    assert built["G"].shape == (n_act, n_act)


def test_build_dm_keys_and_consistency():
    """build_dm returns H, Hpinv, G with Hpinv@H ~ I (well-conditioned coupling
    matrix) and G fused."""
    cfg = _example_config()
    geom = build_geometry(cfg)
    built = dm.build_dm(geom, cfg, reg="tikhonov", mu=1e-6)
    assert set(built) == {"H", "Hpinv", "G"}
    n = built["H"].shape[0]
    # near-perfect inverse for small coupling + tiny mu
    assert np.allclose(built["Hpinv"] @ built["H"], np.eye(n), atol=1e-3)


# ----------------------------------------------------------------------------
# Stroke clipping + units
# ----------------------------------------------------------------------------

def test_stroke_clipping_and_units():
    """apply_stroke_clip respects +/- stroke_max and counts saturated entries;
    to_stroke_units applies gain g (research/05 S4.3, S7)."""
    a = np.array([-2.0, -0.5, 0.0, 0.5, 2.0, 1.0001])
    clipped, n_sat = dm.apply_stroke_clip(a, 1.0)
    assert np.allclose(clipped, [-1.0, -0.5, 0.0, 0.5, 1.0, 1.0])
    # entries with |a| > 1.0 were clipped: -2.0, 2.0, 1.0001 -> 3
    assert n_sat == 3
    # exactly-at-limit values are NOT counted as saturated
    at_limit, n2 = dm.apply_stroke_clip(np.array([1.0, -1.0]), 1.0)
    assert n2 == 0
    # stub alias matches
    c2, n3 = dm.clip_strokes(a, 1.0)
    assert np.allclose(c2, clipped) and n3 == n_sat
    # gain conversion to physical stroke (m)
    z = dm.to_stroke_units(np.array([1.0, -2.0, 0.5]), 1.0e-6)
    assert np.allclose(z, [1.0e-6, -2.0e-6, 0.5e-6])


def test_clip_count_zero_when_within_limits():
    a = np.array([0.1, -0.2, 0.3])
    clipped, n_sat = dm.apply_stroke_clip(a, 1.0)
    assert n_sat == 0
    assert np.allclose(clipped, a)


# ----------------------------------------------------------------------------
# Module-level sanity
# ----------------------------------------------------------------------------

def test_dm_command_matrix_shape_and_sign():
    """dm_command_matrix negates and halves Hpinv (shape preserved)."""
    Hpinv = np.array([[1.0, -0.2], [-0.2, 1.0]])
    G = dm.dm_command_matrix(Hpinv)
    assert G.shape == Hpinv.shape
    assert np.allclose(G, [[-0.5, 0.1], [0.1, -0.5]])
