"""Unit tests for aokit.turbulence (>=7 r0 + >=6 tau0 estimators + combiner).

Skipped pending implementation. Names/docstrings specify intended assertions
(research/04). Validation uses synthetic data with injected ground truth.
"""
import pytest

pytestmark = pytest.mark.skip(reason="TODO(impl): aokit.turbulence not yet implemented")


def test_r0_zernike_variance_recovers_injected():
    """R1: from an ensemble of screens at known r0, fitted r0 is within a few %
    (research/04 R1)."""


def test_r0_slope_variance_independent_of_reconstruction():
    """R2: slope-variance r0 recovers the injected value without reconstruction
    (research/04 R2)."""


def test_r0_dimm_vibration_immune():
    """R3: DIMM r0 is unaffected by an injected common-mode tip/tilt jitter
    (research/04 R3)."""


def test_r0_structure_function_slope_is_5_3():
    """R5: the fitted log-log slope of D_phi(r) is ~5/3, validating Kolmogorov
    (research/04 R5)."""


def test_r0_estimators_agree_within_spread():
    """R1..R7 agree within their spread on a synthetic dataset; the combiner's
    median is close to the injected r0 (research/04 S5)."""


def test_tau0_autocorrelation_recovers_injected():
    """T1: 1/e of the temporal autocorrelation recovers the injected tau0 from a
    frozen-flow series (research/04 T1)."""


def test_tau0_frozen_flow_wind_matches_injected():
    """T5: spatio-temporal cross-correlation recovers the injected wind speed,
    hence tau0 = 0.314 r0/v (research/04 T5)."""


def test_greenwood_closure():
    """T3: tau0 = 0.134 / f_G with f_G = 0.426 v/r0 is consistent with T1/T4
    (research/04 T3, S5)."""
