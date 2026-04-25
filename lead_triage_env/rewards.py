"""Per-step reward shaping (Phase 2 design)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class RewardBreakdown:
    outcome_reward: float
    efficiency_reward: float
    format_compliance: float
    repetition_penalty: float
    budget_compliance: float
    terminal_grader: float

    @property
    def total(self) -> float:
        return (
            self.outcome_reward
            + self.efficiency_reward
            + self.format_compliance
            + self.repetition_penalty
            + self.budget_compliance
            + self.terminal_grader
        )


def _outcome_reward(
    *,
    outcome: str,
    action: str,
    quality: str,
    tier_waste_mult: float,
) -> float:
    if outcome == "ignored_terminal":
        return 0.0

    base = 0.0
    if outcome == "converted":
        base = 10.0
    elif outcome == "positive_reply":
        base = 3.0
    elif outcome == "no_response":
        base = -1.0
    elif outcome == "churned":
        base = -4.0
    elif outcome == "horizon":
        base = -1.5

    step_cost = -0.15
    if outcome == "horizon":
        return base + step_cost

    waste = 0.0
    if quality == "low" and action == "CALL" and outcome in ("no_response", "churned"):
        waste = -2.0 * tier_waste_mult
    if quality == "low" and action == "EMAIL" and outcome in ("no_response", "churned"):
        waste = -0.5 * tier_waste_mult

    return base + waste + step_cost


def reward_breakdown(
    *,
    outcome: str,
    action: str,
    quality: str,
    tier_waste_mult: float,
    converted: bool = False,
    step_index: int = 0,
    max_steps: int = 4,
    action_parsed: bool = True,
    legal_actions: Iterable[str] | None = None,
    repetition_penalty: float = 0.0,
    contact_attempts: int = 0,
    max_contacts: int | None = None,
    terminal_grader: float = 0.0,
) -> RewardBreakdown:
    legal = set(legal_actions or ())
    is_legal = not legal or action in legal
    return RewardBreakdown(
        outcome_reward=_outcome_reward(
            outcome=outcome,
            action=action,
            quality=quality,
            tier_waste_mult=tier_waste_mult,
        ),
        efficiency_reward=0.5 if converted and step_index < (max_steps - 1) else 0.0,
        format_compliance=1.0 if action_parsed and is_legal else 0.0,
        repetition_penalty=repetition_penalty,
        budget_compliance=(
            1.0 if (max_contacts is None or contact_attempts <= max_contacts) else 0.0
        ),
        terminal_grader=terminal_grader,
    )


def step_reward(
    *,
    outcome: str,
    action: str,
    quality: str,
    tier_waste_mult: float,
) -> float:
    """
    Backward-compatible scalar reward for legacy callers.
    """
    return reward_breakdown(
        outcome=outcome,
        action=action,
        quality=quality,
        tier_waste_mult=tier_waste_mult,
    ).outcome_reward


def ignore_opportunity_cost(quality: str) -> float:
    """Extra penalty when IGNORE is chosen on higher-value latent leads."""
    if quality == "high":
        return -1.5
    if quality == "mid":
        return -0.5
    return 0.05
