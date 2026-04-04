"""Transition model: latent quality × action → stochastic outcomes (Phase 2 spec).

Latent quality is one of: low, mid, high (server-side only until exposed for grading).
"""

from __future__ import annotations

import random
from typing import Literal, Tuple

from .task_tier import TierConfig

LeadQuality = Literal["low", "mid", "high"]

# Named outcomes drive rewards and episode termination
LeadOutcome = Literal[
    "none",
    "converted",
    "positive_reply",
    "no_response",
    "churned",
    "ignored_terminal",
]


def _weights_for(quality: LeadQuality, action: str, luck: float) -> Tuple[float, float, float, float]:
    """
    Return relative weights for (converted, positive_reply, no_response, churned).
    IGNORE uses a separate path; FOLLOW_UP uses conditional weights.
    """
    # Base desirability of outcomes (scaled by luck later)
    if action == "IGNORE":
        return (0.0, 0.0, 1.0, 0.0)

    q = {"low": 0, "mid": 1, "high": 2}[quality]
    if action == "CALL":
        base = [(0.04, 0.12, 0.35, 0.49), (0.14, 0.22, 0.28, 0.36), (0.32, 0.28, 0.20, 0.20)][q]
    elif action == "EMAIL":
        base = [(0.02, 0.10, 0.45, 0.43), (0.08, 0.18, 0.39, 0.35), (0.18, 0.26, 0.31, 0.25)][q]
    elif action == "FOLLOW_UP":
        base = [(0.03, 0.11, 0.42, 0.44), (0.12, 0.20, 0.34, 0.34), (0.26, 0.26, 0.24, 0.24)][q]
    else:
        base = (0.0, 0.0, 0.5, 0.5)

    conv, pos, no_r, churn = base
    luck = max(0.25, min(1.75, luck))
    # Luck shifts mass toward better outcomes slightly
    conv *= luck
    pos *= luck
    churn /= luck
    no_r /= (0.85 + 0.15 * luck)
    return conv, pos, no_r, churn


def sample_outcome(quality: LeadQuality, action: str, tier: TierConfig, rng: random.Random) -> LeadOutcome:
    """Single-step stochastic outcome after an action (except IGNORE handled in env)."""
    if action == "IGNORE":
        return "ignored_terminal"

    w = _weights_for(quality, action, tier.outcome_luck)
    conv, pos, no_r, churn = w
    tot = conv + pos + no_r + churn
    if tot <= 0:
        return "no_response"
    r = rng.random() * tot
    if r < conv:
        return "converted"
    r -= conv
    if r < pos:
        return "positive_reply"
    r -= pos
    if r < no_r:
        return "no_response"
    return "churned"
