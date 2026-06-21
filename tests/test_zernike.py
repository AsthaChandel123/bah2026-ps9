"""Unit tests for aokit.zernike (Noll indexing, values, gradients, statistics).

Currently skipped pending implementation (TODO(impl)); the names/docstrings
specify the intended assertions (research/03).
"""
import pytest

pytestmark = pytest.mark.skip(reason="TODO(impl): aokit.zernike not yet implemented")


def test_noll_index_roundtrip():
    """zern_index(noll_from_nm(n,m)) == (n,m) for j=1..36; j=1 -> (0,0) piston;
    j=2 -> (1,1) tip; j=3 -> (1,-1) tilt; j=4 -> (2,0) defocus (research/03 S1.2-1.3)."""


def test_noll_normalisation_rms_unity():
    """Each Noll-normalised Z_j has unit RMS over the unit disc (research/03 S1.1)."""


def test_radial_polynomial_known_values():
    """R_2^0 = 2rho^2-1, R_3^1 = 3rho^3-2rho, etc. (research/03 S1.1)."""


def test_analytic_gradient_matches_finite_difference():
    """zernike_gradient(j) agrees with a numerical gradient of zernike(j)
    within tolerance over the disc (research/03 S2)."""


def test_noll_variance_coefficients():
    """noll_variance(2)=noll_variance(3)~0.448 (tilt); noll_variance(4)~0.0232
    (defocus) (research/03 S3.2, research/04 R1)."""


def test_noll_residual_table():
    """noll_residual(1)~1.0299 (total); noll_residual(3)~0.134 (TT removed)
    (research/03 S3.1)."""
