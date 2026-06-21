"""Unit tests for aokit.reconstructor (zonal Fried + modal Zernike + FTR).

Skipped pending implementation. Names/docstrings specify intended assertions
(research/02, research/03).
"""
import pytest

pytestmark = pytest.mark.skip(reason="TODO(impl): aokit.reconstructor not yet implemented")


def test_fried_gamma_nullspace_piston_and_waffle():
    """Gamma @ 1 == 0 (piston) and Gamma @ waffle == 0 (the Fried waffle null
    mode); rank(Gamma) == N - 2 on the small example (research/02 S4, S6)."""


def test_zonal_reconstructor_removes_waffle():
    """The built R has no waffle content: R @ (Gamma @ waffle) stays ~0 and
    reconstructing a waffle-contaminated slope vector does not blow up
    (research/02 S6, S15)."""


def test_zonal_recovers_known_phase():
    """For a known smooth phase phi_true, s = Gamma @ phi_true, then
    R @ s ~ phi_true (up to piston) within tolerance (research/02 S3)."""


def test_modal_recovers_injected_zernikes():
    """Inject a known Zernike coefficient vector a_true, form s = M @ a_true,
    then M+ @ s ~ a_true; cross-terms ~ 0 (mode purity) (research/03 S2-S3)."""


def test_ftr_matches_dense_reconstructor():
    """The FFT Fried reconstructor agrees with the dense R on a full-grid case
    (cross-check / scaling fallback) (research/02 S7)."""


def test_reconstructor_linearity():
    """Reconstruction is linear: R @ (s1 + s2) == R @ s1 + R @ s2
    (research/07 C.3 superposition test)."""
