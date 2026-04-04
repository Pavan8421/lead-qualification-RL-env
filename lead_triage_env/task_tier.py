"""Task tiers (easy / medium / hard) and per-tier simulator parameters (Phase 2 design)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

TaskTier = Literal["easy", "medium", "hard"]


@dataclass(frozen=True)
class TierConfig:
    """Hyperparameters for lead generation and dynamics."""

    name: TaskTier
    # Observable feature noise (uniform +/- on engagement_score signal)
    engagement_noise: float
    # Scalar applied to non-IGNORE outcome sampling (higher = more forgiving)
    outcome_luck: float
    # Cost multipliers for wasted high-touch on low-quality leads
    waste_penalty_multiplier: float
    # Reference returns for grader normalization (stub; tuned with rollouts in Phase 3)
    grader_oracle_return: float
    grader_random_return: float


TIER_CONFIGS: Dict[TaskTier, TierConfig] = {
    "easy": TierConfig(
        name="easy",
        engagement_noise=0.06,
        outcome_luck=1.15,
        waste_penalty_multiplier=1.0,
        grader_oracle_return=8.5,
        grader_random_return=-2.0,
    ),
    "medium": TierConfig(
        name="medium",
        engagement_noise=0.14,
        outcome_luck=1.0,
        waste_penalty_multiplier=1.25,
        grader_oracle_return=8.5,
        grader_random_return=-2.5,
    ),
    "hard": TierConfig(
        name="hard",
        engagement_noise=0.28,
        outcome_luck=0.88,
        waste_penalty_multiplier=1.55,
        grader_oracle_return=8.5,
        grader_random_return=-3.0,
    ),
}


def normalize_tier(value: str | None, default: TaskTier = "easy") -> TaskTier:
    if value is None:
        return default
    v = str(value).strip().lower()
    if v in TIER_CONFIGS:
        return v  # type: ignore[return-value]
    return default
