"""Unit tests for aokit.reconstructor (zonal Fried + modal Zernike + FTR).

Covers (research/02 zonal, research/03 modal):
  * the Fried interaction matrix Gamma and its piston/waffle null space;
  * the regularized zonal reconstructor R (piston + waffle removed);
  * smooth-phase zonal round-trip and waffle rejection;
  * the modal interaction matrix M, reconstructor Mpinv and mode purity;
  * the phase basis Z (W = Z a);
  * the FFT Fourier-Transform Reconstructor vs the zonal matrix;
  * reconstructor linearity and the unobservable-piston property.

Conventions under test (reconstructor.py / geometry.py docstrings):
  - slope vector is BLOCK layout  s = [sx_1..sx_M, sy_1..sy_M];
  - phase lives on the VALID Fried corner nodes (N_phase = geom.acts.valid.sum());
  - Gamma is (2M, N_phase), R is (N_phase, 2M), M is (2M, J), Mpinv is (J, 2M),
    Z is (N_phase, J).
"""
import os

import numpy as np
import pytest

from aokit.config import load_config
from aokit import geometry as g
from aokit import zernike as z
from aokit import reconstructor as rec


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


@pytest.fixture(scope="module")
def Gamma(geo):
    return rec.build_fried_interaction(geo)


@pytest.fixture(scope="module")
def R(Gamma, geo):
    # Tikhonov with a tiny alpha so the round-trip stays tight, piston + waffle
    # projected out explicitly.
    return rec.build_zonal_reconstructor(
        Gamma, reg="tikhonov", alpha=1e-6,
        remove_piston=True, remove_waffle=True, geom=geo,
    )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _node_coords(geo):
    """Normalized (x, y) unit-disk coordinates of the valid phase nodes."""
    valid_nodes, _ = rec.valid_node_index(geo)
    return geo.acts.x_norm[valid_nodes], geo.acts.y_norm[valid_nodes]


def _demean(a):
    return a - np.mean(a)


def _build_periodic_fried(n, h=1.0):
    """Fully-illuminated *periodic* (toroidal) Fried interaction matrix on an
    ``n x n`` node/cell grid, plus its piston & waffle node vectors.

    On a torus the phase nodes and the slope cells both tile ``n x n``; cell
    ``(i, j)`` uses nodes ``(i,j),(i+1,j),(i,j+1),(i+1,j+1)`` with periodic wrap.
    This makes the dense matrix and the FFT reconstructor exactly comparable
    (no circular-aperture boundary problem).  Returns ``(Gamma, piston, waffle)``
    with ``Gamma`` shape ``(2 n^2, n^2)`` in BLOCK slope layout and node order
    ``flat = row*n + col``.
    """
    N = n * n
    inv2h = 1.0 / (2.0 * h)

    def fidx(i, j):
        return (j % n) * n + (i % n)

    Gx = np.zeros((N, N))
    Gy = np.zeros((N, N))
    k = 0
    for j in range(n):
        for i in range(n):
            a = fidx(i, j)
            b = fidx(i + 1, j)
            c = fidx(i, j + 1)
            d = fidx(i + 1, j + 1)
            Gx[k, b] += inv2h; Gx[k, a] -= inv2h
            Gx[k, d] += inv2h; Gx[k, c] -= inv2h
            Gy[k, c] += inv2h; Gy[k, a] -= inv2h
            Gy[k, d] += inv2h; Gy[k, b] -= inv2h
            k += 1
    Gamma = np.vstack([Gx, Gy])
    ix = np.arange(N) % n
    iy = np.arange(N) // n
    piston = np.ones(N)
    waffle = ((-1.0) ** (ix + iy)).astype(float)
    return Gamma, piston, waffle


# --------------------------------------------------------------------------
# 1. Gamma null space: piston + waffle, rank = N_phase - 2
# --------------------------------------------------------------------------

def test_fried_gamma_nullspace_piston_and_waffle(Gamma, geo):
    """Gamma @ 1 == 0 (piston) and Gamma @ waffle == 0 (the Fried waffle null
    mode); rank(Gamma) == N_phase - 2 (research/02 S4, S6)."""
    valid_nodes, _ = rec.valid_node_index(geo)
    Np = valid_nodes.size
    assert Gamma.shape == (2 * geo.n_sub, Np)

    piston = rec.piston_vector(geo)
    waffle = rec.waffle_vector_nodes(geo)

    assert np.linalg.norm(Gamma @ piston) < 1e-9
    assert np.linalg.norm(Gamma @ waffle) < 1e-9

    rank = np.linalg.matrix_rank(Gamma)
    assert rank == Np - 2

    print(f"\n[zonal] Gamma {Gamma.shape}, N_phase={Np}, rank={rank} (=N_phase-2); "
          f"||G@piston||={np.linalg.norm(Gamma @ piston):.2e}, "
          f"||G@waffle||={np.linalg.norm(Gamma @ waffle):.2e}")


def test_gamma_matches_fried_corner_equations(geo, Gamma):
    """Each Gamma row reproduces the Fried corner-averaging equations exactly
    (research/02 S2.2): s_x=(1/2h)[(b-a)+(d-c)], s_y=(1/2h)[(c-a)+(d-b)]."""
    M = geo.n_sub
    _, remap = rec.valid_node_index(geo)
    h = 1.0
    inv2h = 1.0 / (2.0 * h)
    ci = geo.corner_idx
    for k in range(M):
        a, b, c, d = (int(remap[int(ci[k, 0])]), int(remap[int(ci[k, 1])]),
                      int(remap[int(ci[k, 2])]), int(remap[int(ci[k, 3])]))
        rowx = Gamma[k]
        rowy = Gamma[M + k]
        # x-slope row
        assert rowx[b] == pytest.approx(inv2h)
        assert rowx[a] == pytest.approx(-inv2h)
        assert rowx[d] == pytest.approx(inv2h)
        assert rowx[c] == pytest.approx(-inv2h)
        # y-slope row
        assert rowy[c] == pytest.approx(inv2h)
        assert rowy[a] == pytest.approx(-inv2h)
        assert rowy[d] == pytest.approx(inv2h)
        assert rowy[b] == pytest.approx(-inv2h)


# --------------------------------------------------------------------------
# 2. Zonal reconstructor rejects waffle
# --------------------------------------------------------------------------

def test_zonal_reconstructor_removes_waffle(R, Gamma, geo):
    """R has no waffle content: R @ (Gamma @ waffle) stays ~0 and a
    waffle-contaminated slope vector does not blow up (research/02 S6, S15)."""
    assert R.shape == (Gamma.shape[1], Gamma.shape[0])

    waffle = rec.waffle_vector_nodes(geo)
    piston = rec.piston_vector(geo)

    # Waffle is in Gamma's null space, so its slopes are ~0 ...
    s_waffle = Gamma @ waffle
    assert np.linalg.norm(s_waffle) < 1e-9
    # ... and the reconstructor emits no waffle from them.
    assert np.linalg.norm(R @ s_waffle) < 1e-9

    # R's output is orthogonal to both null modes for ANY slope input.
    rng = np.random.default_rng(0)
    s = rng.standard_normal(Gamma.shape[0])
    phi = R @ s
    assert abs(float(piston @ phi)) < 1e-8 * (np.linalg.norm(phi) + 1.0)
    assert abs(float(waffle @ phi)) < 1e-8 * (np.linalg.norm(phi) + 1.0)

    # Adding waffle to a physical slope vector leaves a bounded reconstruction
    # (no blow-up): the result is identical because waffle is unseen.
    phi_smooth = sum(z.zernike_cartesian(j, *_node_coords(geo)) for j in (2, 3, 4))
    s_phys = Gamma @ phi_smooth
    phi_a = R @ s_phys
    phi_b = R @ (s_phys + 1e3 * s_waffle)       # waffle slopes are ~0 anyway
    assert np.linalg.norm(phi_a - phi_b) < 1e-6 * (np.linalg.norm(phi_a) + 1.0)


# --------------------------------------------------------------------------
# 3. Zonal round-trip on a known smooth phase (piston removed)
# --------------------------------------------------------------------------

def test_zonal_recovers_known_phase(R, Gamma, geo):
    """For a known smooth phase phi_true, s = Gamma @ phi_true, then
    R @ s ~ phi_true up to piston (research/02 S3)."""
    xn, yn = _node_coords(geo)
    # A smooth low-order phase the Fried geometry samples faithfully
    # (tip + tilt + defocus + astigmatism).
    phi_true = (2.0 * z.zernike_cartesian(2, xn, yn)
                + 1.5 * z.zernike_cartesian(3, xn, yn)
                + 1.0 * z.zernike_cartesian(4, xn, yn)
                + 0.7 * z.zernike_cartesian(5, xn, yn)
                + 0.5 * z.zernike_cartesian(6, xn, yn))
    s = Gamma @ phi_true
    phi_hat = R @ s

    a = _demean(phi_true)
    b = _demean(phi_hat)
    rel = np.linalg.norm(a - b) / np.linalg.norm(a)
    print(f"\n[zonal] smooth-phase round-trip rel err = {rel:.3e}")
    assert rel < 0.15


def test_zonal_exact_on_observable_phase(R, Gamma, geo):
    """A phase already in Gamma's row space (piston + waffle projected out) is
    reconstructed essentially exactly: R @ Gamma is the projector onto the
    observable subspace (research/02 S3.1)."""
    xn, yn = _node_coords(geo)
    rng = np.random.default_rng(7)
    phi = sum(c * z.zernike_cartesian(j, xn, yn)
              for c, j in zip(rng.standard_normal(8), range(2, 10)))
    # Remove the unobservable (piston + waffle) part.
    for v in (rec.piston_vector(geo), rec.waffle_vector_nodes(geo)):
        vv = v / np.linalg.norm(v)
        phi = phi - vv * float(vv @ phi)

    phi_hat = R @ (Gamma @ phi)
    rel = np.linalg.norm(phi_hat - phi) / np.linalg.norm(phi)
    print(f"[zonal] observable-phase round-trip rel err = {rel:.3e}")
    assert rel < 1e-2


# --------------------------------------------------------------------------
# 4. Piston is unobservable -> reconstruction has ~zero mean
# --------------------------------------------------------------------------

def test_piston_unobservable_zero_mean(R, Gamma, geo):
    """The reconstructed phase has ~zero mean (piston removed) for any slopes,
    and an added input piston leaves the reconstruction unchanged."""
    xn, yn = _node_coords(geo)
    phi_true = z.zernike_cartesian(4, xn, yn) + 0.5 * z.zernike_cartesian(6, xn, yn)
    s = Gamma @ phi_true
    phi_hat = R @ s
    assert abs(float(np.mean(phi_hat))) < 1e-8 * (np.linalg.norm(phi_hat) + 1.0)

    # A constant offset on phi_true does not change the slopes (Gamma @ 1 == 0),
    # so the reconstruction is identical.
    s_off = Gamma @ (phi_true + 5.0)
    assert np.linalg.norm(s - s_off) < 1e-9
    assert np.allclose(R @ s_off, phi_hat)


# --------------------------------------------------------------------------
# 5. Modal interaction + reconstructor: recover injected Zernike coefficients
# --------------------------------------------------------------------------

def test_modal_recovers_injected_zernikes(geo):
    """Inject a known Zernike coefficient vector a_true, form s = M @ a_true,
    then Mpinv @ s ~ a_true; cross-terms ~0 (mode purity) (research/03 S2-S3)."""
    j_max = 20
    M = rec.build_modal_interaction(geo, j_max, exclude_piston=True)
    Mpinv = rec.build_modal_reconstructor(M, reg="tikhonov", alpha=1e-8)

    J = j_max - 1                                  # modes 2..j_max
    assert M.shape == (2 * geo.n_sub, J)
    assert Mpinv.shape == (J, 2 * geo.n_sub)

    rng = np.random.default_rng(1)
    a_true = rng.standard_normal(J) * 0.1
    s = M @ a_true
    a_hat = Mpinv @ s
    rel = np.linalg.norm(a_hat - a_true) / np.linalg.norm(a_true)
    print(f"\n[modal] J={J}, cond(M)={np.linalg.cond(M):.2f}, "
          f"coeff round-trip rel err = {rel:.3e}")
    assert rel < 1e-6

    # Mode purity: a single unit mode is recovered as a (near) unit vector.
    e = np.zeros(J); e[5] = 1.0
    a_hat_e = Mpinv @ (M @ e)
    assert a_hat_e[5] == pytest.approx(1.0, abs=1e-6)
    off = np.delete(a_hat_e, 5)
    assert np.max(np.abs(off)) < 1e-6


def test_modal_interaction_excludes_piston(geo):
    """Piston (Noll j=1) has zero slope response and is excluded by default."""
    j_max = 10
    M_no_p = rec.build_modal_interaction(geo, j_max, exclude_piston=True)
    M_with_p = rec.build_modal_interaction(geo, j_max, exclude_piston=False)
    assert M_no_p.shape[1] == j_max - 1
    assert M_with_p.shape[1] == j_max
    # The piston column (first, when included) is all zeros.
    assert np.allclose(M_with_p[:, 0], 0.0)


# --------------------------------------------------------------------------
# 6. Phase basis Z: W = Z a, modal reconstruction of the wavefront
# --------------------------------------------------------------------------

def test_phase_basis_synthesizes_wavefront(geo):
    """Z (N_phase x J) sampled at the valid nodes synthesizes the same phase
    that the same Zernike coefficients produce directly (W = Z a)."""
    j_max = 15
    Z = rec.build_phase_basis(geo, j_max, exclude_piston=True)
    J = j_max - 1
    valid_nodes, _ = rec.valid_node_index(geo)
    assert Z.shape == (valid_nodes.size, J)

    xn, yn = _node_coords(geo)
    rng = np.random.default_rng(2)
    a = rng.standard_normal(J)
    W_basis = Z @ a
    W_direct = sum(a[c] * z.zernike_cartesian(j, xn, yn)
                   for c, j in enumerate(range(2, j_max + 1)))
    assert np.allclose(W_basis, W_direct, atol=1e-10)


def test_modal_chain_reconstructs_wavefront(geo):
    """End-to-end modal chain: a known modal phase -> slopes -> Mpinv -> Z gives
    back the phase at the nodes (research/03 S0 full chain)."""
    j_max = 20
    out = rec.build_all(geo, j_max, modal_alpha=1e-8)
    M, Mpinv, Z = out["M"], out["Mpinv"], out["Z"]

    xn, yn = _node_coords(geo)
    rng = np.random.default_rng(11)
    a_true = rng.standard_normal(j_max - 1) * 0.05
    W_true = Z @ a_true                            # phase at nodes from coeffs
    s = M @ a_true                                 # slopes from the same coeffs
    a_hat = Mpinv @ s
    W_hat = Z @ a_hat
    rel = np.linalg.norm(W_hat - W_true) / np.linalg.norm(W_true)
    print(f"[modal] full-chain node-phase rel err = {rel:.3e}")
    assert rel < 1e-6


# --------------------------------------------------------------------------
# 7. FFT Fourier-Transform Reconstructor vs the zonal matrix (full grid)
# --------------------------------------------------------------------------

def test_ftr_matches_dense_reconstructor():
    """On a fully-illuminated (periodic) square grid the FFT Fried reconstructor
    and the dense zonal matrix reconstruct the same smooth phase to machine
    precision (research/02 S7 cross-check / scaling fallback)."""
    n = 16
    h = 1.0
    yy, xx = np.mgrid[0:n, 0:n]
    # A band-limited, exactly-periodic smooth phase (so the FFT periodicity
    # assumption holds and there is no boundary error).
    phi = (np.sin(2 * np.pi * xx / n) + 0.5 * np.cos(2 * np.pi * yy / n)
           + 0.3 * np.sin(2 * np.pi * (xx + yy) / n)
           + 0.2 * np.cos(4 * np.pi * xx / n))

    # Forward Fried slopes on the torus (same convention as the matrix).
    def roll(a, dx, dy):
        return np.roll(np.roll(a, -dy, axis=0), -dx, axis=1)
    a_, b_, c_, d_ = phi, roll(phi, 1, 0), roll(phi, 0, 1), roll(phi, 1, 1)
    sx = (1.0 / (2.0 * h)) * ((b_ - a_) + (d_ - c_))
    sy = (1.0 / (2.0 * h)) * ((c_ - a_) + (d_ - b_))

    # --- FFT reconstructor ---
    phi_ftr = rec.ftr_reconstruct(sx, sy, geometry_type="fried", h=h)

    # --- dense periodic zonal matrix reconstructor ---
    Gamma, piston, waffle = _build_periodic_fried(n, h=h)
    R = rec.build_zonal_reconstructor(
        Gamma, reg="tsvd", remove_piston=True, remove_waffle=True,
        piston=piston, waffle=waffle,
    )
    s = np.concatenate([sx.ravel(), sy.ravel()])
    phi_mat = (R @ s).reshape(n, n)

    A = _demean(phi)
    Bf = _demean(phi_ftr)
    Bm = _demean(phi_mat)

    err_ftr = np.linalg.norm(A - Bf) / np.linalg.norm(A)
    err_mat = np.linalg.norm(A - Bm) / np.linalg.norm(A)
    err_fvm = np.linalg.norm(Bf - Bm) / np.linalg.norm(Bf)
    print(f"\n[FTR] vs-true={err_ftr:.2e}, matrix vs-true={err_mat:.2e}, "
          f"FTR-vs-matrix={err_fvm:.2e}")
    assert err_ftr < 1e-10
    assert err_mat < 1e-10
    assert err_fvm < 1e-10


def test_ftr_filters_zero_nyquist():
    """The Fried FFT filters zero the Nyquist row/col (= waffle removal) and the
    reconstructor leaves zero piston (research/02 S7)."""
    gx, gy = rec.ftr_fried_filters((16, 16))
    assert np.allclose(gx[8, :], 0.0)             # Nyquist row of gx
    assert np.allclose(gy[:, 8], 0.0)             # Nyquist col of gy

    # A pure waffle slope field (which is ~0 for Fried) reconstructs to ~0 phase,
    # and any reconstruction has zero mean (piston pinned).
    rng = np.random.default_rng(5)
    sx = rng.standard_normal((16, 16))
    sy = rng.standard_normal((16, 16))
    phi = rec.ftr_reconstruct(sx, sy)
    assert abs(float(np.mean(phi))) < 1e-9


# --------------------------------------------------------------------------
# 8. Linearity / superposition (research/07 C.3)
# --------------------------------------------------------------------------

def test_reconstructor_linearity(R, geo):
    """Reconstruction is linear: R @ (s1 + s2) == R @ s1 + R @ s2 (and scales)."""
    M2 = R.shape[1]
    rng = np.random.default_rng(3)
    s1 = rng.standard_normal(M2)
    s2 = rng.standard_normal(M2)
    assert np.allclose(R @ (s1 + s2), R @ s1 + R @ s2, atol=1e-9)
    assert np.allclose(R @ (2.5 * s1), 2.5 * (R @ s1), atol=1e-9)


def test_build_all_shapes(geo):
    """build_all returns every matrix with the documented shape."""
    j_max = 20
    out = rec.build_all(geo, j_max)
    M = geo.n_sub
    valid_nodes, _ = rec.valid_node_index(geo)
    Np = valid_nodes.size
    J = j_max - 1
    assert out["Gamma"].shape == (2 * M, Np)
    assert out["R"].shape == (Np, 2 * M)
    assert out["M"].shape == (2 * M, J)
    assert out["Mpinv"].shape == (J, 2 * M)
    assert out["Z"].shape == (Np, J)
