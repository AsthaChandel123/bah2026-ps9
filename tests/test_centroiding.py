"""Unit tests for aokit.centroiding (CoG family + TWCoG).

Skipped pending implementation. Names/docstrings specify intended assertions
(research/01).
"""
import pytest

pytestmark = pytest.mark.skip(reason="TODO(impl): aokit.centroiding not yet implemented")


def test_cog_recovers_known_subpixel_position():
    """A synthetic Gaussian spot at a known sub-pixel (cx,cy) is recovered by
    cog() to < 0.05 px in the noiseless, well-sampled case (research/01 M1)."""


def test_thresholded_cog_suppresses_read_noise():
    """Under read noise, thresholded_cog() has lower centroid variance than
    plain cog() (research/01 M2, S2.1)."""


def test_twcog_near_cramer_rao_at_lab_snr():
    """TWCoG centroid error is within a small factor of the CRLB at lab SNR
    (research/01 S2.3, S12)."""


def test_reference_subtraction_cancels_common_mode_bias():
    """Using the SAME estimator for the reference cancels weighting/threshold
    bias in (centroid - reference) (research/01 S5, S8)."""


def test_correlation_centroid_for_extended_spots():
    """For an elongated/extended spot where CoG is biased, correlation_centroid()
    is more accurate (research/01 M10)."""


def test_gaussfit_is_ground_truth_reference():
    """gaussfit_centroid() matches the injected position closely on a true
    Gaussian spot (offline oracle, research/01 M9)."""
