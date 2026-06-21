"""Unit tests for aokit.dm (influence matrix H, command matrix G, strokes).

Skipped pending implementation. Names/docstrings specify intended assertions
(research/05). The central property: coupling is DECONVOLVED, not naively
sampled.
"""
import pytest

pytestmark = pytest.mark.skip(reason="TODO(impl): aokit.dm not yet implemented")


def test_gaussian_sigma_from_coupling():
    """sigma = d/sqrt(-2 ln c): c=0.15 -> sigma~0.513 d; c=0.135 -> sigma~d/2
    (research/05 S1.3)."""


def test_influence_matrix_is_banded():
    """H is sparse/banded: each surface point sees only nearby actuators
    (research/05 S2)."""


def test_command_matrix_has_anti_coupling_offdiagonals():
    """G (=H+*(-1/2)) contains negative off-diagonal 'anti-coupling' terms; for
    the 1-D 3-actuator example with c=0.15 the recovered commands are SMALLER
    than the naive samples (research/05 S9.1)."""


def test_applied_surface_matches_target():
    """H @ (G_unscaled @ phi) ~ -phi/2: re-applying the commands reproduces the
    conjugate target surface (deconvolution correctness) (research/05 S9.1)."""


def test_naive_sampling_overshoots():
    """Naive a = -phi/2 (no deconvolution) produces H@a that OVERSHOOTS the
    target -- demonstrating why coupling must be inverted (research/05 S3.2, S9.1)."""


def test_stroke_clipping_and_units():
    """clip_strokes respects +/- stroke_max; to_stroke_units applies gain g and
    the factor-of-2 reflection is present (research/05 S4.3, S7)."""


def test_larger_coupling_backs_off_more():
    """With c=0.35 the corrected commands shrink further than with c=0.15
    (research/05 S9.3)."""
