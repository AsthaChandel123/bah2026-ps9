"""Unit tests for aokit.zernike (Noll indexing, values, gradients, statistics).

Conventions tested (research/03 S1-S3, research/04 R1, ARCHITECTURE.md S3.4):
  * Noll single index, j=1 piston; even j -> +m (cos), odd j -> -m (sin).
  * Noll RMS normalisation (unit variance over the unit disc).
  * Analytic Cartesian gradients vs finite differences and vs Gamma matrices.
  * Kolmogorov per-mode variance c_j and Noll residual Delta_J.
"""
import math

import numpy as np
import pytest

import aokit.zernike as Z


# --------------------------------------------------------------------------
# 1. Noll indexing
# --------------------------------------------------------------------------

def test_noll_index_roundtrip():
    """zern_index(noll_from_nm(n,m)) == (n,m) AND noll_from_nm(zern_index(j))==j
    for j=1..36; check the named low orders (research/03 S1.2-1.3)."""
    for j in range(1, 37):
        n, m = Z.zern_index(j)
        assert Z.noll_from_nm(n, m) == j, f"round trip failed at j={j} -> ({n},{m})"

    # Named low-order modes (research/03 Table, S1.3).
    assert Z.zern_index(1) == (0, 0)    # piston
    assert Z.zern_index(2) == (1, 1)    # tip (x-tilt), cos
    assert Z.zern_index(3) == (1, -1)   # tilt (y-tilt), sin
    assert Z.zern_index(4) == (2, 0)    # defocus
    assert Z.zern_index(5) == (2, -2)   # astig oblique (sin)
    assert Z.zern_index(6) == (2, 2)    # astig vertical (cos)
    assert Z.zern_index(7) == (3, -1)   # coma y (sin)
    assert Z.zern_index(8) == (3, 1)    # coma x (cos)
    assert Z.zern_index(11) == (4, 0)   # primary spherical


def test_noll_from_nm_validation():
    """noll_from_nm rejects impossible (n, m) and accepts the full j=1..45 set."""
    with pytest.raises(ValueError):
        Z.noll_from_nm(2, 1)            # n-|m| odd: does not exist
    with pytest.raises(ValueError):
        Z.noll_from_nm(1, 3)            # |m| > n
    # Every Noll index up to 45 must invert consistently.
    for j in range(1, 46):
        n, m = Z.zern_index(j)
        assert Z.noll_from_nm(n, m) == j


# --------------------------------------------------------------------------
# 2. Radial polynomial and closed-form Zernike values
# --------------------------------------------------------------------------

def test_radial_polynomial_known_values():
    """R_2^0 = 2rho^2-1, R_3^1 = 3rho^3-2rho, R_4^0 = 6rho^4-6rho^2+1,
    R_2^2 = rho^2 (research/03 S1.1)."""
    rho = np.linspace(0.0, 1.0, 11)
    assert np.allclose(Z.radial_poly(2, 0, rho), 2 * rho ** 2 - 1)
    assert np.allclose(Z.radial_poly(3, 1, rho), 3 * rho ** 3 - 2 * rho)
    assert np.allclose(Z.radial_poly(4, 0, rho), 6 * rho ** 4 - 6 * rho ** 2 + 1)
    assert np.allclose(Z.radial_poly(2, 2, rho), rho ** 2)
    # n-|m| odd -> identically zero.
    assert np.allclose(Z.radial_poly(3, 0, rho), 0.0)


def test_zernike_closed_forms_cartesian():
    """Z2/Z3 tip/tilt = 2x, 2y; Z4 defocus = sqrt(3)(2(x^2+y^2)-1);
    Z5/Z6 astigmatism; Z8 coma (research/03 Table)."""
    rng = np.random.default_rng(12345)
    x = rng.uniform(-0.7, 0.7, 64)
    y = rng.uniform(-0.7, 0.7, 64)
    rho = np.hypot(x, y)
    inside = rho <= 1.0
    x, y = x[inside], y[inside]
    r2 = x ** 2 + y ** 2

    assert np.allclose(Z.zernike_cartesian(2, x, y), 2 * x)
    assert np.allclose(Z.zernike_cartesian(3, x, y), 2 * y)
    assert np.allclose(Z.zernike_cartesian(4, x, y), math.sqrt(3) * (2 * r2 - 1))
    # Z5 = sqrt6 rho^2 sin(2theta) = sqrt6 * 2xy
    assert np.allclose(Z.zernike_cartesian(5, x, y), math.sqrt(6) * 2 * x * y)
    # Z6 = sqrt6 rho^2 cos(2theta) = sqrt6 (x^2 - y^2)
    assert np.allclose(Z.zernike_cartesian(6, x, y), math.sqrt(6) * (x ** 2 - y ** 2))
    # Z8 = sqrt8 (3rho^3 - 2rho) cos(theta) = sqrt8 (3rho^2 - 2) x
    assert np.allclose(Z.zernike_cartesian(8, x, y),
                       math.sqrt(8) * (3 * r2 - 2) * x)


def test_zernike_zero_outside_pupil():
    """Modes (and gradients) vanish for rho > 1 (the pupil mask)."""
    rho = np.array([1.2, 2.0, 5.0])
    theta = np.array([0.3, 1.1, 2.2])
    for j in (2, 4, 7, 11):
        assert np.allclose(Z.zernike(j, rho, theta), 0.0)
        gx, gy = Z.zernike_gradient(j, rho, theta)
        assert np.allclose(gx, 0.0) and np.allclose(gy, 0.0)


# --------------------------------------------------------------------------
# 3. Noll normalisation / orthonormality
# --------------------------------------------------------------------------

def _disc_quadrature(nr=400, nt=512):
    """Midpoint polar quadrature on the unit disc with weight r dr dtheta / pi
    (so that (1/pi) integral Z_i Z_j dA -> delta_ij)."""
    r = (np.arange(nr) + 0.5) / nr
    t = (np.arange(nt) + 0.5) / nt * 2.0 * np.pi
    R, T = np.meshgrid(r, t, indexing="ij")
    w = (R * (1.0 / nr) * (2.0 * np.pi / nt)) / np.pi
    return R, T, w


def test_noll_normalisation_rms_unity():
    """Each Noll-normalised Z_j has unit RMS over the unit disc (research/03 S1.1)."""
    R, T, w = _disc_quadrature()
    wf = w.ravel()
    for j in range(1, 22):
        zj = Z.zernike(j, R, T).ravel()
        rms2 = np.sum(zj * zj * wf)  # = (1/pi) integral Z_j^2 dA
        assert abs(rms2 - 1.0) < 2e-3, f"mode j={j} variance {rms2}"


def test_orthonormality_matrix():
    """Numerically integrate Z_i Z_j over the unit disc; the Gram matrix is the
    identity (Noll normalisation -> diagonal ~1, off-diagonal ~0)."""
    R, T, w = _disc_quadrature()
    wf = w.ravel()
    modes = list(range(1, 16))
    vals = np.array([Z.zernike(j, R, T).ravel() for j in modes])
    G = (vals * wf) @ vals.T
    err = np.max(np.abs(G - np.eye(len(modes))))
    assert err < 5e-3, f"max |Gram - I| = {err}"


# --------------------------------------------------------------------------
# 4. Analytic gradients
# --------------------------------------------------------------------------

def test_analytic_gradient_matches_finite_difference():
    """zernike_gradient agrees with a central finite difference of zernike over
    the disc for several modes and points (tight tolerance; research/03 S2)."""
    rng = np.random.default_rng(7)
    h = 1e-6
    for j in [2, 3, 4, 5, 6, 7, 8, 11, 12, 16, 21]:
        x = rng.uniform(-0.6, 0.6, 40)
        y = rng.uniform(-0.6, 0.6, 40)
        gx, gy = Z.zernike_gradient_cartesian(j, x, y)
        fdx = (Z.zernike_cartesian(j, x + h, y)
               - Z.zernike_cartesian(j, x - h, y)) / (2 * h)
        fdy = (Z.zernike_cartesian(j, x, y + h)
               - Z.zernike_cartesian(j, x, y - h)) / (2 * h)
        assert np.max(np.abs(gx - fdx)) < 1e-5, f"dZ{j}/dx mismatch"
        assert np.max(np.abs(gy - fdy)) < 1e-5, f"dZ{j}/dy mismatch"


def test_gradient_known_closed_forms():
    """Analytic gradients of the lowest modes match closed forms:
    grad Z2 = (2, 0); grad Z3 = (0, 2); grad Z4 = sqrt3 (4x, 4y)."""
    rng = np.random.default_rng(3)
    x = rng.uniform(-0.5, 0.5, 20)
    y = rng.uniform(-0.5, 0.5, 20)
    gx2, gy2 = Z.zernike_gradient_cartesian(2, x, y)
    assert np.allclose(gx2, 2.0) and np.allclose(gy2, 0.0)
    gx3, gy3 = Z.zernike_gradient_cartesian(3, x, y)
    assert np.allclose(gx3, 0.0) and np.allclose(gy3, 2.0)
    gx4, gy4 = Z.zernike_gradient_cartesian(4, x, y)
    assert np.allclose(gx4, math.sqrt(3) * 4 * x)
    assert np.allclose(gy4, math.sqrt(3) * 4 * y)


def test_gradient_finite_at_origin():
    """The polar gradient is finite at rho=0 (|m|=1 limit handled): grad Z2 at
    the origin is exactly (2, 0)."""
    gx, gy = Z.zernike_gradient(2, np.array([0.0]), np.array([0.0]))
    assert np.allclose(gx, 2.0) and np.allclose(gy, 0.0)
    # Coma (j=8, |m|=1) has zero gradient at the centre (lowest term ~ rho^1 in
    # one direction); just assert it is finite, not NaN/inf.
    gx8, gy8 = Z.zernike_gradient(8, np.array([0.0]), np.array([0.0]))
    assert np.isfinite(gx8).all() and np.isfinite(gy8).all()


def test_gradient_basis_shapes_and_values():
    """zernike_gradient_basis returns (N_pix, N_modes) blocks matching per-mode
    zernike_gradient."""
    rng = np.random.default_rng(9)
    x = rng.uniform(-0.6, 0.6, 50)
    y = rng.uniform(-0.6, 0.6, 50)
    jmax = 10
    Gx, Gy = Z.zernike_gradient_basis(jmax, x, y)
    assert Gx.shape == (50, jmax - 1)   # modes 2..10
    assert Gy.shape == (50, jmax - 1)
    rho = np.hypot(x, y)
    theta = np.arctan2(y, x)
    for c, j in enumerate(range(2, jmax + 1)):
        gx, gy = Z.zernike_gradient(j, rho, theta)
        assert np.allclose(Gx[:, c], gx)
        assert np.allclose(Gy[:, c], gy)


def test_make_gammas_reconstructs_gradient():
    """Noll Gamma matrices reproduce the analytic gradient as a linear
    combination of lower-order Zernikes (research/03 S2a cross-check)."""
    nrad = 4
    g = Z.make_gammas(nrad)
    nmax = (nrad + 1) * (nrad + 2) // 2
    assert g.shape == (2, nmax, nmax)

    rng = np.random.default_rng(2)
    x = rng.uniform(-0.7, 0.7, 60)
    y = rng.uniform(-0.7, 0.7, 60)
    rho = np.hypot(x, y)
    theta = np.arctan2(y, x)
    inside = rho <= 1.0
    basis = np.array([Z.zernike(i, rho, theta) for i in range(1, nmax + 1)])
    for j in range(1, nmax + 1):
        gx_an, gy_an = Z.zernike_gradient(j, rho, theta)
        gx_rec = g[0, :, j - 1] @ basis
        gy_rec = g[1, :, j - 1] @ basis
        assert np.max(np.abs((gx_an - gx_rec)[inside])) < 5e-3
        assert np.max(np.abs((gy_an - gy_rec)[inside])) < 5e-3

    # A couple of exact known Gamma entries.
    assert abs(g[0, 1, 3] - 2 * math.sqrt(3)) < 1e-3   # dZ4/dx -> Z2 coeff 2sqrt3
    assert abs(g[1, 2, 3] - 2 * math.sqrt(3)) < 1e-3   # dZ4/dy -> Z3 coeff 2sqrt3
    assert abs(g[0, 0, 1] - 2.0) < 1e-3                # dZ2/dx -> piston coeff 2


# --------------------------------------------------------------------------
# 5. Basis / synthesis matrices
# --------------------------------------------------------------------------

def test_zernike_basis_matrix():
    """zernike_basis(jmax, x, y) gives (N_pix, N_modes); columns equal the
    per-mode evaluation; mask selects pupil pixels."""
    n = 40
    lin = np.linspace(-1, 1, n)
    xx, yy = np.meshgrid(lin, lin)
    rho = np.hypot(xx, yy)
    mask = rho <= 1.0
    jmax = 10
    B = Z.zernike_basis(jmax, xx, yy, mask=mask)
    assert B.shape == (int(mask.sum()), jmax - 1)   # modes 2..10
    # Column 0 is mode 2 on the masked points.
    expected = Z.zernike(2, rho[mask], np.arctan2(yy[mask], xx[mask]))
    assert np.allclose(B[:, 0], expected)

    # Explicit mode list form preserves order (and can include piston).
    B2 = Z.zernike_basis([4, 2, 1], xx, yy, mask=mask)
    assert B2.shape == (int(mask.sum()), 3)
    assert np.allclose(B2[:, 0], Z.zernike(4, rho[mask],
                                           np.arctan2(yy[mask], xx[mask])))
    assert np.allclose(B2[:, 2], 1.0)               # piston column


def test_zernike_array_orientation():
    """zernike_array returns (n_modes, n_pix) over the disc (modes 2..jmax)."""
    jmax = 8
    ngrid = 32
    A = Z.zernike_array(jmax, ngrid)
    npix = int((np.hypot(*np.meshgrid(np.linspace(-1, 1, ngrid),
                                      np.linspace(-1, 1, ngrid))) <= 1.0).sum())
    assert A.shape == (jmax - 1, npix)


# --------------------------------------------------------------------------
# 6. Kolmogorov statistics: c_j, Delta_J, covariance
# --------------------------------------------------------------------------

def test_noll_variance_coefficients():
    """noll_variance(2)=noll_variance(3)~0.448 (tilt); noll_variance(4)~0.0232
    (defocus); coma ~0.00619 (research/03 S3.2, research/04 R1)."""
    assert abs(Z.noll_variance(2) - 0.448) < 1e-3
    assert abs(Z.noll_variance(3) - 0.448) < 1e-3
    assert abs(Z.noll_variance(4) - 0.0232) < 1e-3
    assert abs(Z.noll_variance(5) - 0.0232) < 1e-3
    assert abs(Z.noll_variance(7) - 0.00619) < 1e-3   # coma
    assert abs(Z.noll_variance(11) - 0.00245) < 1e-3  # spherical
    assert Z.noll_variance(1) == 0.0                  # piston removed


def test_noll_covariance_diagonal_matches_variance():
    """The closed-form Noll covariance diagonal reproduces the per-mode
    coefficients (research/03 S3.2)."""
    # Tilt is the pinned anchor -> exact.
    assert abs(Z.noll_covariance(2, 2) - 0.448) < 1e-9
    # Defocus / coma diagonals from the closed form match the table to ~1e-4.
    assert abs(Z.noll_covariance(4, 4) - 0.0232) < 5e-4
    assert abs(Z.noll_covariance(7, 7) - 0.00619) < 5e-4


def test_noll_covariance_selection_rules():
    """Covariance is zero unless m_j == m_j' and n-n' even and same trig parity."""
    # Different azimuthal order -> 0 (tilt vs defocus).
    assert Z.noll_covariance(2, 4) == 0.0
    # Same m=0 family, n-n' even, both cos -> non-zero (defocus & spherical).
    assert Z.noll_covariance(4, 11) != 0.0
    # The two members of an astigmatism pair (5 sin, 6 cos) are uncorrelated.
    assert Z.noll_covariance(5, 6) == 0.0


def test_noll_residual_table():
    """noll_residual(1)~1.0299 (total); noll_residual(3)~0.134 (TT removed);
    monotone decreasing; asymptotic law beyond the table (research/03 S3.1)."""
    assert abs(Z.noll_residual(1) - 1.0299) < 1e-3
    assert abs(Z.noll_residual(2) - 0.582) < 1e-3
    assert abs(Z.noll_residual(3) - 0.134) < 1e-3
    assert abs(Z.noll_residual(4) - 0.111) < 1e-3
    assert abs(Z.noll_residual(11) - 0.0377) < 1e-3
    assert abs(Z.noll_residual(21) - 0.0208) < 1e-3
    # Strictly decreasing through the tabulated range.
    vals = [Z.noll_residual(j) for j in range(1, 22)]
    assert all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))
    # Asymptotic law for large J: Delta_J ~ 0.2944 J^{-sqrt(3)/2}.
    assert abs(Z.noll_residual(200) - 0.2944 * 200 ** (-math.sqrt(3) / 2)) < 1e-6


def test_residual_consistency_with_variance():
    """Per-mode variance c_j ~ Delta_{j-1} - Delta_j for the low orders
    (the residual drops by each mode's variance as it is corrected)."""
    # tip: Delta_1 - Delta_2 = 1.0299 - 0.582 = 0.448 = c_2.
    assert abs((Z.noll_residual(1) - Z.noll_residual(2)) - Z.noll_variance(2)) < 2e-3
    # tilt: Delta_2 - Delta_3 = 0.582 - 0.134 = 0.448 = c_3.
    assert abs((Z.noll_residual(2) - Z.noll_residual(3)) - Z.noll_variance(3)) < 2e-3
    # defocus: Delta_3 - Delta_4 = 0.134 - 0.111 = 0.023 ~ c_4.
    assert abs((Z.noll_residual(3) - Z.noll_residual(4)) - Z.noll_variance(4)) < 2e-3


# --------------------------------------------------------------------------
# 7. Piston / tip / tilt removal helpers
# --------------------------------------------------------------------------

def test_remove_piston_tip_tilt_vector():
    """remove_piston_tip_tilt drops j=1,2,3 from a Noll-ordered coeff vector."""
    a = np.arange(1.0, 11.0)            # j = 1..10
    a2 = Z.remove_piston_tip_tilt(a)
    assert a2.shape == (7,)
    assert np.allclose(a2, np.arange(4.0, 11.0))   # j = 4..10 remain


def test_remove_modes_basis_columns():
    """remove_modes drops the matching columns of a basis matrix (axis=-1)."""
    rng = np.random.default_rng(5)
    B = rng.normal(size=(15, 6))       # 6 modes -> j = 1..6 columns
    B2 = Z.remove_modes(B, drop=(1, 2, 3), axis=-1)
    assert B2.shape == (15, 3)
    assert np.allclose(B2, B[:, 3:])
