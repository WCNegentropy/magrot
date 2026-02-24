"""Integration tests using validation modules."""

from magrot.validation.zpinch import validate_zpinch
from magrot.validation.theta_pinch import validate_theta_pinch


def test_zpinch_validation():
    checks = validate_zpinch(verbose=False)
    assert all(checks.values()), f"Failed: {[k for k, v in checks.items() if not v]}"


def test_theta_pinch_validation():
    checks = validate_theta_pinch(verbose=False)
    assert all(checks.values()), f"Failed: {[k for k, v in checks.items() if not v]}"
