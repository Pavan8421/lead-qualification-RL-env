"""Map latent lead quality + tier noise → observable features (deterministic RNG)."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple, cast

from .dynamics import LeadQuality
from .task_tier import TierConfig


def _sample_urgency(quality: LeadQuality, tier: TierConfig, rng: random.Random) -> str:
    levels: Tuple[str, str, str] = ("low", "medium", "high")
    if quality == "high":
        weights = (0.14, 0.36, 0.50)
    elif quality == "mid":
        weights = (0.30, 0.48, 0.22)
    else:
        weights = (0.52, 0.38, 0.10)
    if tier.name == "hard" and rng.random() < 0.22:
        return rng.choice(levels)
    r = rng.random()
    acc = 0.0
    for lev, w in zip(levels, weights):
        acc += w
        if r < acc:
            return lev
    return "medium"


_SOURCE_INTENT_BIAS: Dict[str, float] = {
    "linkedin": 0.05,
    "webform": 0.10,
    "referral": 0.20,
    "event": 0.15,
}


def derive_intent_estimate(
    engagement_score: float,
    source: str,
    rng: random.Random,
) -> str:
    """Coarse pre-contact intent bucket from engagement + source, with light noise.

    Returns one of: 'low', 'medium', 'high', 'unknown'.
    Used by env_version=v2 to mask `intent_score` until first CALL/EMAIL.
    """
    bias = _SOURCE_INTENT_BIAS.get(source, 0.0)
    score = 0.7 * float(engagement_score) + 0.3 * bias
    if rng.random() < 0.10:
        return rng.choice(["low", "medium", "high", "unknown"])
    if score < 0.33:
        return "low"
    if score < 0.66:
        return "medium"
    return "high"


def bucket_intent_score(intent_score: float) -> str:
    """Deterministic post-contact bucket of `intent_score`."""
    if intent_score < 0.33:
        return "low"
    if intent_score < 0.66:
        return "medium"
    return "high"


def sample_latent_quality(tier: TierConfig, rng: random.Random) -> LeadQuality:
    """Tier shifts prior over lead quality (easy → more highs)."""
    if tier.name == "easy":
        weights = (0.2, 0.35, 0.45)
    elif tier.name == "medium":
        weights = (0.35, 0.35, 0.30)
    else:
        weights = (0.45, 0.35, 0.20)
    r = rng.random()
    acc = 0.0
    for q, w in zip(("low", "mid", "high"), weights):
        acc += w
        if r < acc:
            return q  # type: ignore[return-value]
    return "low"


def build_observable_features(quality: LeadQuality, tier: TierConfig, rng: random.Random) -> Dict[str, Any]:
    """Construct CRM-style fields correlated with latent quality + observational noise."""
    centers = {
        "low": dict(company_size=40, budget=1500.0, engagement=0.28, days=5),
        "mid": dict(company_size=120, budget=4500.0, engagement=0.52, days=3),
        "high": dict(company_size=280, budget=12000.0, engagement=0.78, days=1),
    }
    c = centers[quality]
    noise = tier.engagement_noise
    engage = max(0.0, min(1.0, c["engagement"] + rng.uniform(-noise, noise)))
    size = max(1, int(c["company_size"] + rng.uniform(-noise * 400, noise * 400)))
    budget = max(0.0, c["budget"] * (1.0 + rng.uniform(-noise * 2, noise * 2)))
    days = max(0, int(c["days"] + rng.uniform(-2, 2)))

    industry = rng.choice(["tech", "finance", "retail", "healthcare", "services"])
    source = rng.choice(["linkedin", "webform", "referral", "event"])

    intent_base = {"low": 0.34, "mid": 0.56, "high": 0.79}[quality]
    intent_score = max(
        0.0, min(1.0, intent_base + rng.uniform(-noise * 1.1, noise * 1.1))
    )

    job_by_quality: Dict[LeadQuality, List[str]] = {
        "low": [
            "Marketing Specialist",
            "Junior Analyst",
            "Sales Rep",
            "Coordinator",
            "Associate",
        ],
        "mid": [
            "Product Manager",
            "Engineering Manager",
            "Sales Manager",
            "Team Lead",
            "IT Manager",
        ],
        "high": [
            "VP Sales",
            "Director of IT",
            "CFO",
            "Head of Procurement",
            "Chief Revenue Officer",
        ],
    }
    job_title = rng.choice(job_by_quality[quality])

    deal_span = {"low": (0.75, 1.35), "mid": (1.0, 2.1), "high": (1.45, 3.6)}[
        quality
    ]
    estimated_deal_value = round(
        budget * rng.uniform(deal_span[0], deal_span[1]), 2
    )

    urgency_level = cast(str, _sample_urgency(quality, tier, rng))

    return dict(
        company_size=size,
        budget=round(budget, 2),
        industry=industry,
        source=source,
        engagement_score=round(engage, 3),
        days_since_contact=days,
        intent_score=round(intent_score, 3),
        job_title=job_title,
        estimated_deal_value=estimated_deal_value,
        urgency_level=urgency_level,
    )
