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


_INDUSTRY_MULTIPLIERS: dict[str, Tuple[float, float, float, float]] = {
    # (conv_mul, pos_mul, no_r_mul, churn_mul)
    "tech": (1.15, 1.10, 1.00, 0.95),
    "finance": (0.85, 0.95, 1.05, 1.10),
    "healthcare": (0.95, 1.00, 1.05, 1.00),
    "retail": (1.00, 1.00, 1.00, 1.00),
    "services": (1.00, 1.00, 1.00, 1.00),
}

_PERSONA_MULTIPLIERS: dict[str, Tuple[float, float, float, float]] = {
    "evaluator_engaged": (1.20, 1.15, 0.95, 0.85),
    "evaluator_stalled": (0.90, 1.00, 1.10, 1.05),
    "champion_internal_blocker": (0.95, 1.00, 1.05, 1.08),
    "tire_kicker": (0.80, 0.95, 1.20, 1.05),
    "ghost": (0.70, 0.85, 1.30, 1.12),
    "re_engaged_after_silence": (1.05, 1.10, 0.95, 0.95),
}


def _weights_for(
    quality: LeadQuality,
    action: str,
    luck: float,
    industry: str | None = None,
    persona: str | None = None,
    argument: str | None = None,
) -> Tuple[float, float, float, float]:
    """
    Return relative weights for (converted, positive_reply, no_response, churned).
    IGNORE uses a separate path; FOLLOW_UP uses conditional weights.
    Industry and persona, when provided, shift the distribution.
    """
    if action == "IGNORE":
        return (0.0, 0.0, 1.0, 0.0)

    q = {"low": 0, "mid": 1, "high": 2}[quality]
    if action == "CALL":
        base = [(0.04, 0.14, 0.40, 0.42), (0.14, 0.24, 0.30, 0.32), (0.32, 0.30, 0.22, 0.16)][q]
    elif action == "EMAIL":
        base = [(0.02, 0.12, 0.48, 0.38), (0.08, 0.20, 0.42, 0.30), (0.18, 0.28, 0.35, 0.19)][q]
    elif action == "FOLLOW_UP":
        base = [(0.03, 0.14, 0.46, 0.37), (0.12, 0.22, 0.37, 0.29), (0.26, 0.30, 0.27, 0.17)][q]
    else:
        base = (0.0, 0.0, 0.5, 0.5)

    conv, pos, no_r, churn = base
    luck = max(0.25, min(1.75, luck))
    conv *= luck
    pos *= luck
    churn /= luck
    no_r /= (0.85 + 0.15 * luck)

    if industry is not None:
        mul = _INDUSTRY_MULTIPLIERS.get(industry)
        if mul is not None:
            conv *= mul[0]
            pos *= mul[1]
            no_r *= mul[2]
            churn *= mul[3]
    if persona is not None:
        mul = _PERSONA_MULTIPLIERS.get(persona)
        if mul is not None:
            conv *= mul[0]
            pos *= mul[1]
            no_r *= mul[2]
            churn *= mul[3]

    # M4 argument-aware shaping (channel + argument effects).
    arg = argument or ""
    if action == "CALL":
        if arg == "closing":
            if quality == "high":
                conv *= 1.25
                churn *= 0.90
            elif quality == "low":
                conv *= 0.75
                churn *= 1.20
        elif arg == "demo":
            pos *= 1.15
            no_r *= 0.95
    elif action == "EMAIL":
        if arg == "case_study":
            pos *= 1.12
            no_r *= 0.94
        elif arg == "value_prop":
            conv *= 1.08
        elif arg == "re_engage":
            pos *= 1.10
            churn *= 0.95
    elif action == "FOLLOW_UP":
        if arg == "call:firm":
            conv *= 1.15
            churn *= 1.05
        elif arg == "call:soft":
            pos *= 1.12
        elif arg == "email:firm":
            churn *= 1.08
        elif arg == "email:soft":
            no_r *= 0.95

    return conv, pos, no_r, churn


def sample_outcome(
    quality: LeadQuality,
    action: str,
    tier: TierConfig,
    rng: random.Random,
    industry: str | None = None,
    persona: str | None = None,
    argument: str | None = None,
) -> LeadOutcome:
    """Single-step stochastic outcome after an action (except IGNORE handled in env)."""
    if action == "IGNORE":
        return "ignored_terminal"

    w = _weights_for(
        quality,
        action,
        tier.outcome_luck,
        industry=industry,
        persona=persona,
        argument=argument,
    )
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
