"""Task tiers (easy / medium / hard) and per-tier simulator parameters (Phase 2 design)."""

from __future__ import annotations

from dataclasses import dataclass
import random
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


def recompute_tier_anchors(
    *,
    episodes_per_tier: int = 2000,
    seed: int = 1337,
) -> Dict[TaskTier, Dict[str, float]]:
    """Estimate oracle/random returns after M4 action-space expansion.

    Oracle has access to latent quality and picks an action+argument pair with
    highest expected one-step reward under the current simulator.
    Random samples uniformly from the expanded action space.
    """
    # Local imports avoid import cycle with dynamics -> task_tier.
    from .dynamics import sample_outcome
    from .rewards import step_reward
    rng = random.Random(seed)
    expanded_actions = [
        ("EMAIL", "generic"),
        ("EMAIL", "value_prop"),
        ("EMAIL", "case_study"),
        ("EMAIL", "re_engage"),
        ("CALL", "discovery"),
        ("CALL", "demo"),
        ("CALL", "closing"),
        ("FOLLOW_UP", "email:soft"),
        ("FOLLOW_UP", "email:firm"),
        ("FOLLOW_UP", "call:soft"),
        ("FOLLOW_UP", "call:firm"),
        ("IGNORE", ""),
    ]
    quality_weights_by_tier = {
        "easy": [("low", 0.2), ("mid", 0.35), ("high", 0.45)],
        "medium": [("low", 0.35), ("mid", 0.35), ("high", 0.30)],
        "hard": [("low", 0.45), ("mid", 0.35), ("high", 0.20)],
    }
    industries = ["tech", "finance", "retail", "healthcare", "services"]

    def sample_quality(tier: TaskTier) -> str:
        r = rng.random()
        acc = 0.0
        for q, w in quality_weights_by_tier[tier]:
            acc += w
            if r <= acc:
                return q
        return "mid"

    results: Dict[TaskTier, Dict[str, float]] = {}
    for tier, cfg in TIER_CONFIGS.items():
        random_returns: list[float] = []
        oracle_returns: list[float] = []
        for _ in range(episodes_per_tier):
            quality = sample_quality(tier)
            industry = rng.choice(industries)

            # Random baseline
            ch, arg = expanded_actions[rng.randrange(len(expanded_actions))]
            out = sample_outcome(
                quality=quality,  # type: ignore[arg-type]
                action=ch,
                tier=cfg,
                rng=rng,
                industry=industry,
                argument=arg,
            )
            random_returns.append(
                step_reward(
                    outcome=out,
                    action=ch,
                    quality=quality,  # type: ignore[arg-type]
                    tier_waste_mult=cfg.waste_penalty_multiplier,
                )
            )

            # Oracle baseline via brute-force expected one-step score proxy
            best = -1e9
            for och, oarg in expanded_actions:
                score = 0.0
                for _ in range(6):
                    o = sample_outcome(
                        quality=quality,  # type: ignore[arg-type]
                        action=och,
                        tier=cfg,
                        rng=rng,
                        industry=industry,
                        argument=oarg,
                    )
                    score += step_reward(
                        outcome=o,
                        action=och,
                        quality=quality,  # type: ignore[arg-type]
                        tier_waste_mult=cfg.waste_penalty_multiplier,
                    )
                best = max(best, score / 6.0)
            oracle_returns.append(best)

        results[tier] = {
            "grader_random_return": sum(random_returns) / max(1, len(random_returns)),
            "grader_oracle_return": sum(oracle_returns) / max(1, len(oracle_returns)),
        }
    return results
