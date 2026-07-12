"""Build the explainable PASS/FAIL decision from evaluated quality checks."""

from __future__ import annotations

from datascience.schemas import CheckStatus, QualityCheck, QualityDecision


def build_decision(checks: list[QualityCheck]) -> QualityDecision:
    """PASS only if no check FAILed; None if nothing was verifiable at all.

    NOT_AVAILABLE checks do not fail the product but stay visible in the
    report so the operator knows what could not be verified. If every check
    is NOT_AVAILABLE/SKIPPED (e.g. no product/picture to actually inspect),
    there is nothing to base a PASS/FAIL judgement on, so overall_pass stays
    None instead of defaulting to PASS.
    """
    failures = [c for c in checks if c.status == CheckStatus.FAIL]
    verifiable = [c for c in checks if c.status in (CheckStatus.PASS, CheckStatus.FAIL)]

    failure_reasons = [
        f"{c.name}: {c.reason or 'out of specification'}"
        + (f" (measured {c.measured}, expected {c.expected})" if c.measured else "")
        for c in failures
    ]

    overall_pass = (len(failures) == 0) if verifiable else None

    return QualityDecision(
        overall_pass=overall_pass,
        checks=checks,
        failure_reasons=failure_reasons,
    )
