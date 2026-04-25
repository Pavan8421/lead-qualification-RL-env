"""Persona-driven pre-history generation for v2 observations."""

from __future__ import annotations

import random
from typing import Literal

from .models import HistorySummary, Interaction, LeadEvent

PersonaName = Literal[
    "evaluator_engaged",
    "evaluator_stalled",
    "champion_internal_blocker",
    "tire_kicker",
    "ghost",
    "re_engaged_after_silence",
]

_PERSONA_TOPICS: dict[PersonaName, list[str]] = {
    "evaluator_engaged": ["integration scope", "security review", "demo recap"],
    "evaluator_stalled": ["budget review", "procurement queue", "timing concerns"],
    "champion_internal_blocker": ["internal buy-in", "legal approval", "security sign-off"],
    "tire_kicker": ["feature list", "pricing curiosity", "general information"],
    "ghost": ["initial outreach", "follow-up ping", "checking in"],
    "re_engaged_after_silence": ["re-engagement note", "new quarter priorities", "renewed interest"],
}

_PERSONA_OBJECTIONS: dict[PersonaName, list[str]] = {
    "evaluator_engaged": ["needs case study"],
    "evaluator_stalled": ["budget freeze", "priority shift"],
    "champion_internal_blocker": ["security review pending", "legal delay"],
    "tire_kicker": ["no urgent need"],
    "ghost": ["non-responsive stakeholder"],
    "re_engaged_after_silence": ["timeline uncertainty"],
}


def sample_persona(latent_quality: str, rng: random.Random) -> PersonaName:
    """Sample a persona conditioned loosely on latent quality."""
    if latent_quality == "high":
        weights = [
            ("evaluator_engaged", 0.28),
            ("champion_internal_blocker", 0.22),
            ("re_engaged_after_silence", 0.18),
            ("evaluator_stalled", 0.14),
            ("tire_kicker", 0.10),
            ("ghost", 0.08),
        ]
    elif latent_quality == "mid":
        weights = [
            ("evaluator_stalled", 0.24),
            ("champion_internal_blocker", 0.20),
            ("tire_kicker", 0.18),
            ("re_engaged_after_silence", 0.16),
            ("evaluator_engaged", 0.14),
            ("ghost", 0.08),
        ]
    else:
        weights = [
            ("ghost", 0.24),
            ("tire_kicker", 0.24),
            ("evaluator_stalled", 0.20),
            ("re_engaged_after_silence", 0.14),
            ("champion_internal_blocker", 0.10),
            ("evaluator_engaged", 0.08),
        ]
    r = rng.random()
    acc = 0.0
    for name, prob in weights:
        acc += prob
        if r <= acc:
            return name
    return "evaluator_stalled"


def _event_for(persona: PersonaName, direction: str, rng: random.Random) -> LeadEvent:
    if direction == "inbound":
        if persona in ("evaluator_engaged", "re_engaged_after_silence"):
            return rng.choices(
                ["positive_reply", "no_response"],
                weights=[0.8, 0.2],
                k=1,
            )[0]
        if persona == "ghost":
            return "no_response"
        return rng.choices(["positive_reply", "no_response"], weights=[0.45, 0.55], k=1)[0]
    if persona == "ghost":
        return "no_response"
    if persona == "tire_kicker":
        return rng.choices(["positive_reply", "no_response"], weights=[0.3, 0.7], k=1)[0]
    return rng.choices(["positive_reply", "no_response"], weights=[0.5, 0.5], k=1)[0]


def sample_history(persona: PersonaName, rng: random.Random) -> list[Interaction]:
    """Generate synthetic prior touches (3-10) spanning recent weeks."""
    n = rng.randint(3, 10)
    history: list[Interaction] = []
    day = -rng.randint(10, 30)
    for i in range(n):
        gap = rng.randint(1, 6)
        day += gap
        channel = rng.choice(["CALL", "EMAIL", "FOLLOW_UP"])
        direction = "inbound" if rng.random() < 0.35 else "outbound"
        outcome = _event_for(persona, direction, rng)
        sentiment = rng.uniform(-0.15, 0.85) if outcome == "positive_reply" else rng.uniform(-0.7, 0.2)
        duration = rng.randint(3, 25) if channel == "CALL" else rng.randint(1, 8)
        topic = rng.choice(_PERSONA_TOPICS[persona])
        note = f"{topic}; persona={persona}; touch={i+1}"
        history.append(
            Interaction(
                day_offset=day,
                channel=channel,  # type: ignore[arg-type]
                direction=direction,  # type: ignore[arg-type]
                outcome=outcome,
                topic=topic,
                sentiment=round(sentiment, 3),
                duration_min=duration,
                note=note,
            )
        )
    return history


def summarize_history(history: list[Interaction], persona: PersonaName | None = None) -> HistorySummary:
    if not history:
        return HistorySummary()

    ordered = sorted(history, key=lambda h: h.day_offset)
    inbound = [h for h in ordered if h.direction == "inbound"]
    last_inbound_sentiment = inbound[-1].sentiment if inbound else 0.0

    max_gap = 0
    for i in range(1, len(ordered)):
        max_gap = max(max_gap, ordered[i].day_offset - ordered[i - 1].day_offset)

    objections = []
    if persona is not None:
        objections = list(_PERSONA_OBJECTIONS.get(persona, []))

    return HistorySummary(
        total_touches=len(ordered),
        days_since_first_touch=max(0, -ordered[0].day_offset),
        days_since_last_touch=max(0, -ordered[-1].day_offset),
        inbound_count=len(inbound),
        last_inbound_sentiment=round(float(last_inbound_sentiment), 3),
        longest_silence_gap=max_gap,
        raised_objections=objections,
    )
