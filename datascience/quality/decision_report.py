"""Build the explainable PASS/FAIL decision from evaluated quality checks."""

from __future__ import annotations

from datascience.schemas import CheckStatus, QualityCheck, QualityDecision


def build_decision(checks: list[QualityCheck]) -> QualityDecision:
    """PASS only if no check FAILed.

    NOT_AVAILABLE checks do not fail the product but stay visible in the
    report so the operator knows what could not be verified.
    """
    failures = [c for c in checks if c.status == CheckStatus.FAIL]

    failure_reasons = [
        f"{c.name}: {c.reason or 'out of specification'}"
        + (f" (measured {c.measured}, expected {c.expected})" if c.measured else "")
        for c in failures
    ]

    return QualityDecision(
        overall_pass=len(failures) == 0,
        checks=checks,
        failure_reasons=failure_reasons,
    )
