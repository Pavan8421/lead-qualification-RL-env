"""M3c — Industry-aware outcome bias tests."""

from __future__ import annotations

import random

from lead_triage_env.dynamics import sample_outcome
from lead_triage_env.task_tier import TIER_CONFIGS


def _conversion_rate(industry: str, n: int = 4000, seed: int = 1234) -> float:
    rng = random.Random(seed)
    tier = TIER_CONFIGS["easy"]
    converted = 0
    for _ in range(n):
        out = sample_outcome("high", "CALL", tier, rng, industry=industry)
        if out == "converted":
            converted += 1
    return converted / n


def test_tech_converts_more_than_finance() -> None:
    tech_rate = _conversion_rate("tech")
    finance_rate = _conversion_rate("finance")
    assert tech_rate > finance_rate, (tech_rate, finance_rate)


def test_neutral_industries_close_to_baseline() -> None:
    baseline = _conversion_rate("retail")
    services = _conversion_rate("services")
    assert abs(services - baseline) < 0.05


def test_sample_outcome_without_industry_is_backward_compatible() -> None:
    rng = random.Random(0)
    tier = TIER_CONFIGS["easy"]
    out = sample_outcome("mid", "EMAIL", tier, rng)
    assert out in {"converted", "positive_reply", "no_response", "churned"}
