"""Unit tests for aokit.geometry (Fried sub-aperture & actuator grids).

Validates the canonical orderings and geometric relations that the C real-time
core and the Python reconstructor both rely on:

  * lenslet count and a sensible valid-sub-aperture count;
  * the Fried (n+1)x(n+1) actuator-corner grid (4 distinct corners per valid
    sub-aperture; forward/inverse adjacency consistent);
  * reference-spot count and spacing (~ lenslet pitch in pixels);
  * the px -> slope scaling (exact: pixel_size / focal_length);
  * the circular pupil mask (symmetric, roughly circular).

Conventions under test (geometry.py module docstring):
  - slope vector is BLOCK layout  s = [sx_1..sx_M, sy_1..sy_M];
  - corners (a,b,c,d) = (TL,TR,BL,BR) on the row-major (n+1)x(n+1) corner grid.
"""
import os

import numpy as np
import pytest

from aokit.config import load_config
from aokit import geometry as g


CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "example_config.json",
)


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def geo(cfg):
    return g.build_geometry(cfg)


# --------------------------------------------------------------------------
# 1. Sub-aperture grid: lenslet count and valid-sub-aperture count
# --------------------------------------------------------------------------

def test_lenslet_and_valid_subaperture_count(cfg, geo):
    """N lenslets per the config; valid-subap count is sensible (>0, <= N*N)."""
    nx, ny = cfg.mla.n_lenslets_x, cfg.mla.n_lenslets_y
    assert (nx, ny) == (10, 10)
    n_nominal = nx * ny
    assert cfg.n_sub_nominal == n_nominal

    M = geo.n_sub
    assert 0 < M <= n_nominal
    # All canonical-order arrays agree on M.
    assert geo.subaps.ref_x.shape[0] == M
    assert geo.subaps.ref_y.shape[0] == M
    assert geo.subaps.ij.shape == (M, 2)
    assert geo.n_slopes == 2 * M

    # Print for the report (visible with `pytest -s`).
    print(f"\n[geometry] valid sub-apertures = {M} of {n_nominal} "
          f"(slope vector length 2M = {geo.n_slopes})")


def test_subaperture_grid_within_frame_and_ordering(cfg, geo):
    """Lenslets are enumerated row-major; valid subset keeps that order."""
    ij = geo.subaps.ij
    nx = cfg.mla.n_lenslets_x
    flat = ij[:, 1] * nx + ij[:, 0]          # row*nx + col
    # Strictly increasing => canonical row-major order preserved after masking.
    assert np.all(np.diff(flat) > 0)
    # Indices in range.
    assert ij[:, 0].min() >= 0 and ij[:, 0].max() < cfg.mla.n_lenslets_x
    assert ij[:, 1].min() >= 0 and ij[:, 1].max() < cfg.mla.n_lenslets_y


# --------------------------------------------------------------------------
# 2. Fried actuator grid: (n+1)x(n+1) nodes, 4 distinct corners, adjacency
# --------------------------------------------------------------------------

def test_actuator_grid_node_count(cfg, geo):
    """Fried actuator grid has exactly (n+1)x(n+1) nodes."""
    nx, ny = cfg.mla.n_lenslets_x, cfg.mla.n_lenslets_y
    assert geo.n_act == (nx + 1) * (ny + 1)
    assert geo.acts.x.shape == (geo.n_act,)
    assert geo.acts.y.shape == (geo.n_act,)
    assert geo.acts.ij.shape == (geo.n_act, 2)
    # Config's nominal actuator count agrees (Fried: n_act == n_lenslets + 1).
    assert cfg.n_act_nominal == geo.n_act


def test_each_valid_subaperture_has_four_distinct_corners(geo):
    """Each valid sub-aperture maps to exactly 4 distinct corner actuators."""
    ci = geo.corner_idx
    M = geo.n_sub
    assert ci.shape == (M, 4)
    for k in range(M):
        corners = ci[k]
        assert len(np.unique(corners)) == 4
        assert corners.min() >= 0 and corners.max() < geo.n_act


def test_adjacency_forward_inverse_consistent(geo):
    """lenslet->actuator and actuator->lenslet maps are mutually consistent."""
    fwd = geo.lenslet_to_act          # (M, 4)
    inv = geo.act_to_lenslet          # (n_act, 4), -1 padded
    assert fwd.shape == (geo.n_sub, 4)
    assert inv.shape == (geo.n_act, 4)

    # Every (lenslet k, actuator a) in forward appears in inverse[a].
    for k in range(geo.n_sub):
        for a in fwd[k]:
            assert k in inv[int(a)]

    # And every non-pad entry in inverse appears in the corresponding forward row.
    for a in range(geo.n_act):
        for k in inv[a]:
            if k >= 0:
                assert a in fwd[int(k)]

    # Each valid actuator touches between 1 and 4 valid sub-apertures.
    counts = (inv >= 0).sum(axis=1)
    touched = counts > 0
    assert np.all(counts[touched] >= 1)
    assert np.all(counts <= 4)
    # The valid-actuator mask equals "touches >=1 valid sub-aperture".
    assert np.array_equal(geo.acts.valid, touched)


def test_corner_indices_match_fried_worked_example():
    """fried_corner_indices(2,2) matches the research/02 S4 worked example."""
    ci = g.fried_corner_indices(2, 2)
    expected = np.array([
        [0, 1, 3, 4],   # C0 (a,b,c,d)=(TL,TR,BL,BR)
        [1, 2, 4, 5],   # C1
        [3, 4, 6, 7],   # C2
        [4, 5, 7, 8],   # C3
    ], dtype=np.int64)
    assert np.array_equal(ci, expected)


def test_corner_geometry_matches_subaperture_centers(cfg, geo):
    """A sub-aperture center is the mean of its 4 corner-actuator positions."""
    ax, ay = geo.acts.x, geo.acts.y
    ci = geo.corner_idx
    cx = ax[ci].mean(axis=1)
    cy = ay[ci].mean(axis=1)
    assert np.allclose(cx, geo.subaps.ref_x, atol=1e-6)
    assert np.allclose(cy, geo.subaps.ref_y, atol=1e-6)


# --------------------------------------------------------------------------
# 3. Reference spots: count and spacing ~ lenslet pitch in pixels
# --------------------------------------------------------------------------

def test_reference_spot_count(geo):
    """Reference-spot count equals the number of valid sub-apertures."""
    assert geo.subaps.ref_x.shape[0] == geo.n_sub
    assert geo.subaps.ref_y.shape[0] == geo.n_sub


def test_reference_spot_spacing_is_lenslet_pitch(cfg, geo):
    """Adjacent reference spots are spaced ~ pitch_m / pixel_size (px)."""
    ppl = cfg.px_per_lenslet                      # pitch_m / pixel_size_m
    assert ppl == pytest.approx(cfg.mla.pitch_m / cfg.camera.pixel_size_m)

    ref = {(int(c), int(r)): (geo.subaps.ref_x[i], geo.subaps.ref_y[i])
           for i, (c, r) in enumerate(geo.subaps.ij)}

    dx, dy = [], []
    for (c, r), (x, y) in ref.items():
        if (c + 1, r) in ref:                      # right neighbour
            dx.append(ref[(c + 1, r)][0] - x)
        if (c, r + 1) in ref:                      # bottom neighbour
            dy.append(ref[(c, r + 1)][1] - y)

    assert dx and dy
    assert np.allclose(dx, ppl, atol=1e-6)
    assert np.allclose(dy, ppl, atol=1e-6)


def test_reference_registration_from_flat_frame(cfg):
    """register_references recovers spot positions from a synthetic flat frame."""
    full = g.build_subaperture_grid(cfg, valid_only=False)
    H, W = cfg.camera.frame_h, cfg.camera.frame_w
    yy, xx = np.mgrid[0:H, 0:W]
    frame = np.zeros((H, W), dtype=np.float64)

    # Place a compact Gaussian spot at every fully-on-frame nominal reference.
    truth = {}
    for k in range(full.ref_x.shape[0]):
        cx, cy = full.ref_x[k], full.ref_y[k]
        if 3.0 <= cx < W - 3.0 and 3.0 <= cy < H - 3.0 and full.valid[k]:
            frame += 120.0 * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 1.8 ** 2)))
            truth[k] = (cx, cy)

    reg = g.register_references(cfg, frame, full)
    # Where we placed a clean, well-separated spot the CoG matches to << 1 px.
    for k, (cx, cy) in truth.items():
        assert reg.ref_x[k] == pytest.approx(cx, abs=0.1)
        assert reg.ref_y[k] == pytest.approx(cy, abs=0.1)

    # The flux-based active mask agrees with the pupil-geometry valid count.
    mask = g.active_subaperture_mask(cfg, frame, reg, flux_frac=0.5)
    assert int(mask.sum()) == int(full.valid.sum())


def test_set_references_override(cfg, geo):
    """set_references overrides centroids without touching ordering/windows."""
    new = g.set_references(geo.subaps,
                           geo.subaps.ref_x + 0.5, geo.subaps.ref_y - 0.5)
    assert np.allclose(new.ref_x - geo.subaps.ref_x, 0.5)
    assert np.allclose(new.ref_y - geo.subaps.ref_y, -0.5)
    assert np.array_equal(new.ij, geo.subaps.ij)
    assert new.w == geo.subaps.w and new.h == geo.subaps.h
    with pytest.raises(ValueError):
        g.set_references(geo.subaps, geo.subaps.ref_x[:-1], geo.subaps.ref_y[:-1])


# --------------------------------------------------------------------------
# 4. Slope scaling: 1-px displacement -> pixel_size/focal_length radians
# --------------------------------------------------------------------------

def test_one_pixel_displacement_to_slope_exact(cfg):
    """A 1-pixel displacement converts to exactly pixel_size/focal_length rad."""
    expected = cfg.camera.pixel_size_m / cfg.mla.focal_length_m
    assert g.displacement_to_slope(cfg, 1.0) == expected
    assert cfg.slope_scale == expected
    # Vectorized form is exact too.
    disp = np.array([0.0, 1.0, -2.0, 3.5])
    assert np.array_equal(g.displacement_to_slope(cfg, disp), disp * expected)


def test_slopes_from_centroids_block_layout(cfg, geo):
    """slopes_from_centroids uses the canonical block layout [sx.., sy..]."""
    M = geo.n_sub
    rng = np.random.default_rng(0)
    dx = rng.standard_normal(M)
    dy = rng.standard_normal(M)
    cx = geo.subaps.ref_x + dx
    cy = geo.subaps.ref_y + dy

    s = g.slopes_from_centroids(cfg, cx, cy, geo.subaps.ref_x, geo.subaps.ref_y)
    assert s.shape == (2 * M,)

    lay = g.slope_vector_layout(M)
    assert lay["layout"] == "block"
    xs, xe = lay["x_slice"]
    ys, ye = lay["y_slice"]
    assert (xs, xe, ys, ye) == (0, M, M, 2 * M)
    scale = cfg.slope_scale
    assert np.allclose(s[xs:xe], dx * scale)
    assert np.allclose(s[ys:ye], dy * scale)


# --------------------------------------------------------------------------
# 5. Pupil mask: symmetric and roughly circular (area ~ pi/4 * fraction)
# --------------------------------------------------------------------------

def test_pupil_mask_symmetric_and_circular(cfg):
    """Pupil mask is L-R / U-D symmetric and has area ~ pi/4 of the array."""
    for n in (32, 64, 65):
        m = g.pupil_mask(cfg, n)
        assert m.shape == (n, n)
        assert m.dtype == bool
        # Symmetry.
        assert np.array_equal(m, m[:, ::-1])      # left-right
        assert np.array_equal(m, m[::-1, :])      # up-down
        # Area of an inscribed disk over a square is pi/4 (~0.785).
        frac = m.mean()
        assert frac == pytest.approx(np.pi / 4, abs=0.03)


def test_pupil_mask_is_connected_disk(cfg):
    """Mask is a single filled disk: every row's True pixels are contiguous."""
    m = g.pupil_mask(cfg, 64)
    for row in m:
        idx = np.where(row)[0]
        if idx.size:
            # contiguous run => max-min+1 == count
            assert idx.max() - idx.min() + 1 == idx.size


# --------------------------------------------------------------------------
# 6. Normalized pupil sampling (for Zernike basis / gradients)
# --------------------------------------------------------------------------

def test_normalized_coordinates_on_unit_disk(cfg, geo):
    """Sub-aperture and actuator normalized coords are sensible unit-disk values."""
    # Sub-aperture centers: radius <= ~1 (a few near the edge may sit just past).
    r_sub = np.hypot(geo.subap_x_norm, geo.subap_y_norm)
    assert r_sub.max() <= 1.2
    # Center actuator (col, row) = (nx/2, ny/2) for even n lands at the origin.
    r_act = np.hypot(geo.acts.x_norm, geo.acts.y_norm)
    assert r_act.min() == pytest.approx(0.0, abs=1e-9)
    # Normalization is consistent with the pupil radius in pixels.
    R = g.pupil_radius_px(cfg)
    assert R == pytest.approx(0.5 * cfg.pupil.diameter_m / cfg.camera.pixel_size_m)
    cx0 = cfg.pupil.center_x_px
    assert np.allclose(geo.acts.x_norm, (geo.acts.x - cx0) / R)
