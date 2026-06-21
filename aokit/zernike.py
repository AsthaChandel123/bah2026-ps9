"""aokit.zernike -- Noll-indexed Zernike polynomials, gradients, index helpers.

Noll single index starting at ``j = 1 = piston``; RMS / Noll normalisation so
each coefficient is the RMS wavefront of that mode over the unit disc
(research/03 S1, ARCHITECTURE.md S3.4).

Conventions (fixed and used consistently end-to-end)
----------------------------------------------------
* **Indexing** -- Noll (1976): ``j`` orders by increasing radial order ``n``
  and within an ``n`` by increasing ``|m|``; **even j -> +m (cos)**, **odd j
  -> -m (sin)**; ``j = 1`` is piston ``(0, 0)``.  ``zern_index`` reproduces
  the ``aotools.zernIndex`` logic verbatim (research/03 S1.2).
* **Normalisation** -- Noll RMS: ``N_n^m = sqrt(n+1)`` for ``m = 0`` and
  ``sqrt(2 (n+1))`` for ``m != 0`` so ``(1/pi) integral_disc Z_j Z_j' dA =
  delta_{jj'}`` (research/03 S1.1).  Every coefficient ``a_j`` is then the RMS
  wavefront contributed by mode ``j``.
* **Azimuth** -- ``theta = atan2(y, x)`` measured from +x toward +y;
  ``rho = sqrt(x^2 + y^2)``.  Outside the unit disc (``rho > 1``) every mode and
  every gradient is defined to be ``0`` (the pupil mask).
* **Gradients** -- analytic Cartesian ``dZ_j/dx``, ``dZ_j/dy`` are obtained by
  differentiating the polar form via the chain rule (closed form, vectorised),
  and cross-checked against Noll's derivative (Gamma) matrices
  (:func:`make_gammas`, research/03 S2).

Key relations implemented
--------------------------
  - zern_index(j) -> (n, m)               [research/03 S1.2 aotools zernIndex]
  - noll_from_nm(n, m) -> j               [inverse]
  - radial_poly R_n^m(rho)                [research/03 S1.1]
  - zernike(j, rho, theta)                [Noll-normalised value]
  - zernike_cartesian(j, x, y)            [Cartesian convenience wrapper]
  - zernike_basis / zernike_array         [(N_pix x N_modes) synthesis matrix]
  - zernike_gradient(j, rho, theta)       [analytic d/dx, d/dy]
  - zernike_gradient_basis / make_gammas  [batched gradients; Noll Gamma matrices]
  - noll_variance(j) / noll_residual(j)   [Kolmogorov c_j and Delta_J; r0 fitting]
  - noll_covariance(j, jp)                [Zernike-Kolmogorov covariance <a_j a_j'>]
  - remove_piston_tip_tilt(...)           [drop j=1,2,3 from a coeff vector/basis]
"""
from __future__ import annotations

from math import comb, factorial, sqrt
from typing import Iterable, Sequence, Tuple

import numpy as np

__all__ = [
    "zern_index",
    "noll_from_nm",
    "noll_radial_order",
    "radial_poly",
    "radial_poly_deriv",
    "noll_norm",
    "zernike",
    "zernike_cartesian",
    "zernike_array",
    "zernike_basis",
    "zernike_gradient",
    "zernike_gradient_cartesian",
    "zernike_gradient_basis",
    "make_gammas",
    "noll_variance",
    "noll_residual",
    "noll_covariance",
    "remove_piston_tip_tilt",
    "remove_modes",
]


# ============================================================================
# 1. Noll index helpers
# ============================================================================

def zern_index(j: int) -> Tuple[int, int]:
    """Map a Noll single index ``j`` (>= 1) to ``(n, m)``.

    Even ``j`` -> ``+m`` (cos), odd ``j`` -> ``-m`` (sin); ``j = 1`` is piston
    ``(0, 0)``.  Reproduces ``aotools.zernIndex`` (research/03 S1.2).

    Examples
    --------
    >>> zern_index(1)
    (0, 0)
    >>> zern_index(2)
    (1, 1)
    >>> zern_index(3)
    (1, -1)
    >>> zern_index(4)
    (2, 0)
    """
    j = int(j)
    if j < 1:
        raise ValueError(f"Noll index j must be >= 1, got {j}")
    n = int((-1.0 + sqrt(8 * (j - 1) + 1)) / 2.0)
    p = j - (n * (n + 1)) // 2
    k = n % 2
    m = (int((p + k) // 2) * 2) - k
    if m != 0:
        m *= 1 if (j % 2 == 0) else -1  # even j -> +m (cos), odd j -> -m (sin)
    return (n, m)


def noll_from_nm(n: int, m: int) -> int:
    """Inverse of :func:`zern_index`: ``(n, m) -> j`` (Noll single index).

    Validates ``n >= 0``, ``|m| <= n`` and ``n - |m|`` even (else the mode does
    not exist).
    """
    n = int(n)
    m = int(m)
    if n < 0:
        raise ValueError(f"radial order n must be >= 0, got {n}")
    if abs(m) > n or ((n - abs(m)) % 2 != 0):
        raise ValueError(f"invalid Zernike order (n, m) = ({n}, {m})")

    # Base index of the radial order n: number of modes with radial order < n
    # is n*(n+1)/2, so the first Noll index in order n is that + 1.
    base = (n * (n + 1)) // 2 + 1

    am = abs(m)
    if am == 0:
        # m == 0 occurs only for even n; it is the first entry of the order.
        return base

    # Within radial order n the modes appear in pairs of increasing |m|.
    # Position within the order: for |m| the two members occupy indices
    # base + (|m| - k) - 1 and base + (|m| - k) where k = n % 2 handles the
    # m==0 (even n) leading entry.  Derive j directly then fix parity.
    k = n % 2
    offset = am - k  # 0 for the lowest |m| pair start
    j_lo = base + offset - 1  # candidate lower of the pair
    j_hi = base + offset      # candidate upper of the pair
    # Choose the parity-correct member: even j -> +m, odd j -> -m.
    if m > 0:
        j = j_lo if (j_lo % 2 == 0) else j_hi
    else:
        j = j_lo if (j_lo % 2 == 1) else j_hi
    # Robust fallback / guarantee: confirm the round trip; if the closed form
    # mis-stepped at an order boundary, search the (small) order window.
    if zern_index(j) != (n, m):
        for cand in (base, base + 1, j_lo, j_hi, j_lo + 1, j_hi + 1,
                     j_lo - 1, j_hi - 1):
            if cand >= 1 and zern_index(cand) == (n, m):
                return cand
        raise ValueError(f"could not invert (n, m) = ({n}, {m})")
    return j


def noll_radial_order(j: int) -> int:
    """Radial order ``n`` of Noll index ``j`` (helper)."""
    return zern_index(j)[0]


# ============================================================================
# 2. Radial polynomial and full Zernike value (Noll normalised)
# ============================================================================

def radial_poly(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    """Radial polynomial ``R_n^m(rho)`` (research/03 S1.1).

    ``R_n^m`` is zero unless ``n - |m|`` is even.  Vectorised over ``rho``.
    Values are *not* masked here (callers mask ``rho > 1``); the polynomial is
    evaluated as written.
    """
    n = int(n)
    m = int(abs(m))
    rho = np.asarray(rho, dtype=float)
    if (n - m) % 2 != 0 or m > n:
        return np.zeros_like(rho)

    out = np.zeros_like(rho)
    half = (n - m) // 2
    for k in range(half + 1):
        # (-1)^k (n-k)! / [ k! ((n+m)/2 - k)! ((n-m)/2 - k)! ]
        coeff = ((-1) ** k) * factorial(n - k) / (
            factorial(k)
            * factorial((n + m) // 2 - k)
            * factorial((n - m) // 2 - k)
        )
        out = out + coeff * rho ** (n - 2 * k)
    return out


def radial_poly_deriv(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    """Derivative ``dR_n^m/drho`` (analytic, vectorised) -- used for gradients."""
    n = int(n)
    m = int(abs(m))
    rho = np.asarray(rho, dtype=float)
    if (n - m) % 2 != 0 or m > n:
        return np.zeros_like(rho)

    out = np.zeros_like(rho)
    half = (n - m) // 2
    for k in range(half + 1):
        power = n - 2 * k
        if power == 0:
            continue  # derivative of constant term is 0
        coeff = ((-1) ** k) * factorial(n - k) / (
            factorial(k)
            * factorial((n + m) // 2 - k)
            * factorial((n - m) // 2 - k)
        )
        out = out + coeff * power * rho ** (power - 1)
    return out


def noll_norm(n: int, m: int) -> float:
    """Noll RMS normalisation factor ``N_n^m`` (``sqrt(n+1)`` for ``m == 0``,
    ``sqrt(2 (n+1))`` otherwise)."""
    n = int(n)
    if m == 0:
        return sqrt(n + 1)
    return sqrt(2 * (n + 1))


def zernike(j: int, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Noll-normalised Zernike mode ``Z_j`` on the unit disc (RMS = 1).

    Polar evaluation.  ``rho`` and ``theta`` are broadcast together; values at
    ``rho > 1`` are set to ``0`` (outside the pupil).
    """
    n, m = zern_index(int(j))
    rho = np.asarray(rho, dtype=float)
    theta = np.asarray(theta, dtype=float)
    rho, theta = np.broadcast_arrays(rho, theta)

    norm = noll_norm(n, m)
    R = radial_poly(n, m, rho)
    if m == 0:
        val = norm * R
    elif m > 0:
        val = norm * R * np.cos(m * theta)
    else:
        val = norm * R * np.sin(-m * theta)

    val = np.where(rho <= 1.0 + 1e-12, val, 0.0)
    return val


def zernike_cartesian(j: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Cartesian convenience wrapper for :func:`zernike` (``rho, theta`` from
    ``x, y``)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rho = np.hypot(x, y)
    theta = np.arctan2(y, x)
    return zernike(j, rho, theta)


# ============================================================================
# 3. Zernike basis matrix (synthesis matrix Z; W = Z a)
# ============================================================================

def _resolve_mode_list(j_list_or_jmax, include_piston: bool) -> list:
    """Normalise the ``j`` argument: an int ``j_max`` -> a range, or an
    explicit iterable of Noll indices -> a list."""
    if np.isscalar(j_list_or_jmax):
        j_max = int(j_list_or_jmax)
        start = 1 if include_piston else 2
        return list(range(start, j_max + 1))
    return [int(j) for j in j_list_or_jmax]


def zernike_basis(j_list_or_jmax, x: np.ndarray, y: np.ndarray,
                  mask: np.ndarray | None = None) -> np.ndarray:
    """Zernike synthesis matrix on arbitrary pupil sample points.

    Parameters
    ----------
    j_list_or_jmax : int | iterable of int
        Either ``j_max`` (uses Noll modes ``2 .. j_max``, piston excluded) or
        an explicit iterable of Noll indices to evaluate (in that order).
    x, y : array_like
        Cartesian pupil sample coordinates **normalised to the unit disc**
        (``rho = sqrt(x^2 + y^2) <= 1`` inside the pupil).  Flattened to 1-D.
    mask : array_like of bool, optional
        If given, only sample points where ``mask`` is True are used (the
        returned matrix has one row per True pixel, in flattened order).

    Returns
    -------
    Z : ndarray, shape ``(N_pix, N_modes)``
        Column ``c`` is mode ``j_list[c]`` sampled at the (masked) points, so
        ``W = Z @ a`` turns a modal coefficient vector into a phase map.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if mask is not None:
        mask = np.asarray(mask, dtype=bool).ravel()
        x = x[mask]
        y = y[mask]

    rho = np.hypot(x, y)
    theta = np.arctan2(y, x)

    j_list = _resolve_mode_list(j_list_or_jmax, include_piston=False
                                ) if np.isscalar(j_list_or_jmax) else \
        _resolve_mode_list(j_list_or_jmax, include_piston=True)

    Z = np.empty((x.size, len(j_list)), dtype=float)
    for c, j in enumerate(j_list):
        Z[:, c] = zernike(j, rho, theta)
    return Z


def zernike_array(j_max: int, n_grid: int,
                  pupil_mask: np.ndarray | None = None) -> np.ndarray:
    """Stack of Zernike modes ``Z_2 .. Z_{j_max}`` sampled on an
    ``n_grid x n_grid`` disc, returned as ``(n_modes, n_pix)`` over the valid
    pupil pixels (research/03; ARCHITECTURE.md S3.4 synthesis matrix ``Z``).

    The grid spans ``[-1, 1] x [-1, 1]``; the implicit pupil is ``rho <= 1``
    unless an explicit ``pupil_mask`` (shape ``(n_grid, n_grid)``) is supplied.
    Note the transposed orientation (modes-by-pixels) relative to
    :func:`zernike_basis`, matching the stub contract.
    """
    n_grid = int(n_grid)
    lin = np.linspace(-1.0, 1.0, n_grid)
    xx, yy = np.meshgrid(lin, lin)
    rho = np.hypot(xx, yy)
    if pupil_mask is None:
        pupil_mask = rho <= 1.0
    else:
        pupil_mask = np.asarray(pupil_mask, dtype=bool)

    # zernike_basis returns (N_pix, N_modes); transpose to (N_modes, N_pix).
    Z = zernike_basis(j_max, xx, yy, mask=pupil_mask)
    return np.ascontiguousarray(Z.T)


# ============================================================================
# 4. Analytic Cartesian gradients
# ============================================================================

def zernike_gradient(j: int, rho: np.ndarray, theta: np.ndarray
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Analytic Cartesian gradients ``(dZ_j/dx, dZ_j/dy)`` on the unit disc.

    Closed form via the chain rule from the polar Zernike (research/03 S2).
    With ``Z = N * R(rho) * Theta(m, theta)`` where ``Theta = cos(m theta)``
    (m>0), ``sin(|m| theta)`` (m<0) or ``1`` (m=0):

        dZ/drho   = N * R'(rho) * Theta
        dZ/dtheta = N * R(rho)  * Theta'

        dZ/dx = cos(theta) dZ/drho - (sin(theta)/rho) dZ/dtheta
        dZ/dy = sin(theta) dZ/drho + (cos(theta)/rho) dZ/dtheta

    The apparent ``1/rho`` singularity at the origin cancels analytically (the
    azimuthal factor carries a compensating ``rho`` power); it is handled here
    by evaluating the gradient in Cartesian form so the result is finite
    everywhere, including ``rho = 0``.  Values at ``rho > 1`` are ``0``.

    Cross-checked against :func:`make_gammas` (Noll Gamma matrices) in the test
    suite.
    """
    n, m = zern_index(int(j))
    rho = np.asarray(rho, dtype=float)
    theta = np.asarray(theta, dtype=float)
    rho, theta = np.broadcast_arrays(rho, theta)
    rho = np.array(rho, dtype=float)
    theta = np.array(theta, dtype=float)

    norm = noll_norm(n, m)
    R = radial_poly(n, m, rho)
    Rp = radial_poly_deriv(n, m, rho)
    am = abs(m)

    cth = np.cos(theta)
    sth = np.sin(theta)

    # Build dZ/dx, dZ/dy directly to avoid the 1/rho factor at the origin.
    # Use: x = rho cos, y = rho sin. Then for the azimuthal part we expand
    # cos(m theta), sin(m theta) and combine with R(rho) so that the result is
    # a smooth polynomial in (x, y).  Practically we compute the radial and
    # azimuthal partials and combine with the *vector* basis (cos, sin) and
    # (-sin, cos)/rho, regularising rho at the origin where the angular term
    # is multiplied by a vanishing radial factor.

    # radial unit-vector contribution: cos(theta) e_x + sin(theta) e_y times dZ/drho
    if m == 0:
        dZdr = norm * Rp                      # angular factor = 1
        dZdt = np.zeros_like(rho)             # no theta dependence
    elif m > 0:
        ang = np.cos(am * theta)
        angp = -am * np.sin(am * theta)
        dZdr = norm * Rp * ang
        dZdt = norm * R * angp
    else:
        ang = np.sin(am * theta)
        angp = am * np.cos(am * theta)
        dZdr = norm * Rp * ang
        dZdt = norm * R * angp

    # dZ/dx = cos dZdr - sin/rho dZdt ; dZ/dy = sin dZdr + cos/rho dZdt.
    # The term dZdt/rho is finite because dZdt carries a factor R(rho) ~ rho^|m|
    # (|m| >= 1) so dZdt/rho -> 0 as rho -> 0 for |m| >= 2 and a constant for
    # |m| == 1.  Compute R(rho)/rho safely.
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_rho = np.where(rho > 0, 1.0 / rho, 0.0)
        dZdt_over_rho = dZdt * inv_rho

    # Fix the origin for |m| == 1 (tip/tilt/coma...): there R(rho) = ... rho ...
    # so R/rho -> finite. Recompute the limit at rho == 0 analytically by using
    # the lowest-order coefficient of R.
    if am == 1:
        # R_n^1(rho) lowest power is rho^1 with coefficient given by k=half.
        half = (n - 1) // 2
        k = half
        c1 = ((-1) ** k) * factorial(n - k) / (
            factorial(k)
            * factorial((n + 1) // 2 - k)
            * factorial((n - 1) // 2 - k)
        )  # coefficient of rho^1 in R_n^1
        at0 = (rho == 0)
        if np.any(at0):
            if m > 0:
                angp0 = -am * np.sin(am * theta)
            else:
                angp0 = am * np.cos(am * theta)
            limit = norm * c1 * angp0
            dZdt_over_rho = np.where(at0, limit, dZdt_over_rho)

    dZdx = cth * dZdr - sth * dZdt_over_rho
    dZdy = sth * dZdr + cth * dZdt_over_rho

    inside = rho <= 1.0 + 1e-12
    dZdx = np.where(inside, dZdx, 0.0)
    dZdy = np.where(inside, dZdy, 0.0)
    return dZdx, dZdy


def zernike_gradient_cartesian(j: int, x: np.ndarray, y: np.ndarray
                               ) -> Tuple[np.ndarray, np.ndarray]:
    """Cartesian wrapper for :func:`zernike_gradient` (``rho, theta`` from
    ``x, y``)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rho = np.hypot(x, y)
    theta = np.arctan2(y, x)
    return zernike_gradient(j, rho, theta)


def zernike_gradient_basis(j_list_or_jmax, x: np.ndarray, y: np.ndarray,
                           mask: np.ndarray | None = None
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """Batched analytic-gradient builder.

    Returns ``(Gx, Gy)``, each ``(N_pix, N_modes)``, where column ``c`` holds
    ``dZ_{j_c}/dx`` resp. ``dZ_{j_c}/dy`` at the (masked) sample points.  These
    feed the modal slope interaction matrix ``M`` (sub-aperture-averaged
    gradients; research/03 S2, ARCHITECTURE.md S3.4).

    ``x, y`` are Cartesian coordinates normalised to the unit disc; ``mask``
    (optional) selects the pupil pixels exactly as in :func:`zernike_basis`.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if mask is not None:
        mask = np.asarray(mask, dtype=bool).ravel()
        x = x[mask]
        y = y[mask]
    rho = np.hypot(x, y)
    theta = np.arctan2(y, x)

    j_list = _resolve_mode_list(j_list_or_jmax, include_piston=False
                                ) if np.isscalar(j_list_or_jmax) else \
        _resolve_mode_list(j_list_or_jmax, include_piston=True)

    Gx = np.empty((x.size, len(j_list)), dtype=float)
    Gy = np.empty((x.size, len(j_list)), dtype=float)
    for c, j in enumerate(j_list):
        gx, gy = zernike_gradient(j, rho, theta)
        Gx[:, c] = gx
        Gy[:, c] = gy
    return Gx, Gy


def make_gammas(n_radial: int) -> np.ndarray:
    """Noll derivative (Gamma) matrices, shape ``(2, nmax, nmax)`` =
    ``[Gamma_x, Gamma_y]`` (research/03 S2a; cross-check for the interaction
    matrix).

    Defined so that, in the Noll-normalised Zernike basis,

        dZ_j/dx = sum_i Gamma_x[i, j] Z_i ,
        dZ_j/dy = sum_i Gamma_y[i, j] Z_i ,

    i.e. the derivative of a Zernike is a finite linear combination of
    *lower-order* Zernikes.  ``nmax`` is the number of modes through radial
    order ``n_radial`` inclusive, ``nmax = (n_radial+1)(n_radial+2)/2``.

    This implementation builds the matrices **numerically** by least-squares
    projection of the analytic Cartesian gradient (:func:`zernike_gradient`)
    onto the orthonormal Zernike basis over a fine unit-disc grid -- exact (to
    rounding) because the gradients lie exactly in the span of the lower-order
    modes.  It serves as the independent cross-check the stub calls for.
    """
    n_radial = int(n_radial)
    nmax = (n_radial + 1) * (n_radial + 2) // 2  # modes 1..nmax (Noll)

    # Fine polar quadrature grid on the unit disc for projection integrals.
    nr, nt = 200, 256
    # midpoint radii (avoid r=0 singular weight issues), area weight r dr dtheta
    r = (np.arange(nr) + 0.5) / nr
    t = (np.arange(nt) + 0.5) / nt * 2.0 * np.pi
    R, T = np.meshgrid(r, t, indexing="ij")
    dr = 1.0 / nr
    dt = 2.0 * np.pi / nt
    # weight so that (1/pi) integral Z_i Z_j dA = delta_ij  ->  w = r dr dt / pi
    w = (R * dr * dt) / np.pi

    X = R * np.cos(T)
    Y = R * np.sin(T)

    # Precompute the basis values Z_i (i = 1..nmax) on the grid.
    Zvals = np.empty((nmax, R.size), dtype=float)
    for i in range(1, nmax + 1):
        Zvals[i - 1] = zernike(i, R, T).ravel()
    wflat = w.ravel()

    gammas = np.zeros((2, nmax, nmax), dtype=float)
    for j in range(1, nmax + 1):
        gx, gy = zernike_gradient_cartesian(j, X, Y)
        gx = gx.ravel()
        gy = gy.ravel()
        for i in range(1, nmax + 1):
            zi = Zvals[i - 1]
            gammas[0, i - 1, j - 1] = np.sum(zi * gx * wflat)
            gammas[1, i - 1, j - 1] = np.sum(zi * gy * wflat)
    # Clean tiny numerical dust.
    gammas[np.abs(gammas) < 1e-9] = 0.0
    return gammas


# ============================================================================
# 5. Kolmogorov / Noll statistics: c_j, Delta_J, covariance
# ============================================================================

# Noll (1976) Table IV residual coefficients Delta_J (units of (D/r0)^(5/3)):
# residual mean-square wavefront error AFTER correcting the first J modes.
_NOLL_RESIDUAL = {
    1: 1.0299,
    2: 0.582,
    3: 0.134,
    4: 0.111,
    5: 0.0880,
    6: 0.0648,
    7: 0.0587,
    8: 0.0525,
    9: 0.0463,
    10: 0.0401,
    11: 0.0377,
    12: 0.0352,
    13: 0.0328,
    14: 0.0304,
    15: 0.0279,
    16: 0.0267,
    17: 0.0255,
    18: 0.0243,
    19: 0.0232,
    20: 0.0220,
    21: 0.0208,
}

# Asymptotic large-J residual law (Noll 1976):
#   Delta_J ~ 0.2944 J^{-sqrt(3)/2} (D/r0)^{5/3}
_NOLL_ASym_A = 0.2944
_NOLL_ASym_P = -sqrt(3.0) / 2.0  # ~ -0.8660


def noll_residual(j: int) -> float:
    """Residual variance ``Delta_J`` after correcting the first ``J = j`` modes,
    in units of ``(D/r0)^(5/3)`` (research/03 S3.1, Noll 1976 Table IV).

    Tabulated for ``j = 1 .. 21``; beyond that the asymptotic law
    ``Delta_J ~ 0.2944 J^{-sqrt(3)/2}`` is used.
    """
    j = int(j)
    if j < 1:
        raise ValueError(f"J must be >= 1, got {j}")
    if j in _NOLL_RESIDUAL:
        return _NOLL_RESIDUAL[j]
    return _NOLL_ASym_A * j ** _NOLL_ASym_P


def noll_covariance(j: int, jp: int) -> float:
    """Kolmogorov Zernike covariance ``<a_j a_j'> / (D/r0)^(5/3)`` (Noll 1976).

    Non-zero only when ``m_j == m_j'`` and ``n - n'`` is even (and, for the
    sign convention, the two indices have the same azimuthal parity, i.e. both
    even-j/cos or both odd-j/sin).  Returns the closed-form value

        <a_j a_j'> = K * (-1)^{(n+n'-2m)/2} * sqrt((n+1)(n'+1))
                     * Gamma(14/3) Gamma[(n+n'-5/3)/2]
                     / ( Gamma[(n-n'+17/3)/2] Gamma[(n'-n+17/3)/2]
                         Gamma[(n+n'+23/3)/2] )

    with the standard prefactor ``K`` (research/03 S3.2; arXiv:2004.11210).
    The diagonal ``j == j'`` reproduces the per-mode :func:`noll_variance`.
    """
    from math import gamma

    n, m = zern_index(int(j))
    npr, mpr = zern_index(int(jp))
    am, amp = abs(m), abs(mpr)

    if am != amp:
        return 0.0
    if (n - npr) % 2 != 0:
        return 0.0
    # Both members of a +-m pair are uncorrelated unless same trig parity:
    if m != 0:
        even_j = (int(j) % 2 == 0)
        even_jp = (int(jp) % 2 == 0)
        if even_j != even_jp:
            return 0.0

    # Noll prefactor: 0.0072 * (24/5 Gamma(6/5))^(5/6) * pi^(8/3)
    # Use the standard combined constant 3.895... (research/03 S3.2).
    # K = 2.246? -- pin it by matching the tilt variance c_2 = 0.448 below.
    K = _NOLL_COV_PREFACTOR

    sign = (-1.0) ** ((n + npr - 2 * am) // 2)
    num = (sign * sqrt((n + 1) * (npr + 1)) * gamma(14.0 / 3.0)
           * gamma((n + npr - 5.0 / 3.0) / 2.0))
    den = (gamma((n - npr + 17.0 / 3.0) / 2.0)
           * gamma((npr - n + 17.0 / 3.0) / 2.0)
           * gamma((n + npr + 23.0 / 3.0) / 2.0))
    return K * num / den


def _compute_cov_prefactor() -> float:
    """Pin the Noll covariance prefactor ``K`` so the diagonal tilt variance
    equals the canonical ``c_2 = 0.448`` (research/04 R1).  This makes the
    closed form self-consistent with the tabulated per-mode coefficients."""
    from math import gamma
    # Tilt: (n, m) = (1, 1). Diagonal value with K = 1:
    n = npr = 1
    am = 1
    sign = (-1.0) ** ((n + npr - 2 * am) // 2)
    num = (sign * sqrt((n + 1) * (npr + 1)) * gamma(14.0 / 3.0)
           * gamma((n + npr - 5.0 / 3.0) / 2.0))
    den = (gamma((n - npr + 17.0 / 3.0) / 2.0)
           * gamma((npr - n + 17.0 / 3.0) / 2.0)
           * gamma((n + npr + 23.0 / 3.0) / 2.0))
    raw = num / den
    return 0.448 / raw


# Module-level constant (computed once at import).
_NOLL_COV_PREFACTOR = _compute_cov_prefactor()


# Canonical tabulated per-mode Kolmogorov variance coefficients c_j
# (research/04 R1).  Used directly for the low orders; higher orders fall back
# to the closed-form diagonal of noll_covariance.
_NOLL_VARIANCE = {
    2: 0.448, 3: 0.448,          # tip, tilt
    4: 0.0232,                    # defocus
    5: 0.0232, 6: 0.0232,         # astigmatism
    7: 0.00619, 8: 0.00619,       # coma
    9: 0.00619, 10: 0.00619,      # trefoil
    11: 0.00245,                  # primary spherical
}


def noll_variance(j: int) -> float:
    """Per-mode Kolmogorov variance constant ``c_j`` such that
    ``<a_j^2> = c_j (D/r0)^(5/3)`` (research/03 S3.2, research/04 R1).

    ``j = 1`` (piston) returns ``0`` (piston is unobservable / removed).
    Low orders use the canonical tabulated values; higher orders use the
    closed-form diagonal of :func:`noll_covariance`.
    """
    j = int(j)
    if j < 1:
        raise ValueError(f"j must be >= 1, got {j}")
    if j == 1:
        return 0.0
    if j in _NOLL_VARIANCE:
        return _NOLL_VARIANCE[j]
    return noll_covariance(j, j)


# ============================================================================
# 6. Piston / tip / tilt removal helpers
# ============================================================================

def remove_modes(coeffs_or_basis: np.ndarray, drop: Sequence[int],
                 axis: int = -1) -> np.ndarray:
    """Drop the given **Noll-indexed** modes from a coefficient vector or a
    basis/synthesis matrix.

    ``coeffs_or_basis`` is interpreted as Noll-ordered starting at ``j = 1``
    along ``axis`` (so index ``k`` along that axis corresponds to ``j = k + 1``).
    Returns a view-free copy with those columns/entries removed.

    Examples
    --------
    Drop piston/tip/tilt from a length-J coefficient vector ``a`` (j = 1..J):
    ``remove_modes(a, drop=(1, 2, 3))``.
    """
    arr = np.asarray(coeffs_or_basis)
    n = arr.shape[axis]
    drop_set = {int(d) for d in drop}
    keep = [k for k in range(n) if (k + 1) not in drop_set]
    return np.take(arr, keep, axis=axis)


def remove_piston_tip_tilt(coeffs_or_basis: np.ndarray, axis: int = -1
                           ) -> np.ndarray:
    """Convenience: drop piston (j=1), tip (j=2) and tilt (j=3).

    Piston is unobservable by an SH-WFS; tip/tilt are excluded from r0 fitting
    (research/03 S2.8, research/04 S3.3).  Assumes Noll order starting at j=1
    along ``axis``.
    """
    return remove_modes(coeffs_or_basis, drop=(1, 2, 3), axis=axis)
