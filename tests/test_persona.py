"""M3d persona/history tests."""

from __future__ import annotations

import os
import random

from lead_triage_env.models import LeadTriageAction
from lead_triage_env.persona import (
    sample_history,
    sample_persona,
    summarize_history,
)
from lead_triage_env.server.lead_triage_environment import LeadTriageEnvironment


def test_sample_persona_returns_known_archetype() -> None:
    rng = random.Random(7)
    persona = sample_persona("mid", rng)
    assert persona in {
        "evaluator_engaged",
        "evaluator_stalled",
        "champion_internal_blocker",
        "tire_kicker",
        "ghost",
        "re_engaged_after_silence",
    }


def test_sample_history_size_and_summary_fields() -> None:
    rng = random.Random(11)
    history = sample_history("evaluator_stalled", rng)
    assert 3 <= len(history) <= 10
    summary = summarize_history(history, persona="evaluator_stalled")
    assert summary.total_touches == len(history)
    assert summary.days_since_first_touch >= summary.days_since_last_touch
    assert summary.inbound_count >= 0
    assert isinstance(summary.raised_objections, list)


def test_v2_reset_exposes_history_not_persona_or_latent() -> None:
    os.environ["LEAD_ENV_VERSION"] = "v2"
    env = LeadTriageEnvironment()
    obs = env.reset(seed=123, task_tier="easy")
    dump = obs.model_dump(mode="json")
    assert 3 <= len(obs.contact_history) <= 10
    assert obs.history_summary.total_touches == len(obs.contact_history)
    assert "latent_quality" not in dump
    assert "persona_name" not in dump


def test_v2_step_appends_history_and_caps_to_ten() -> None:
    os.environ["LEAD_ENV_VERSION"] = "v2"
    env = LeadTriageEnvironment(max_steps=20)
    obs = env.reset(seed=321, task_tier="easy")
    start_len = len(obs.contact_history)

    # Force enough steps to exercise capping behavior.
    for _ in range(8):
        legal = set(obs.legal_actions)
        action = "EMAIL" if "EMAIL" in legal else ("CALL" if "CALL" in legal else "IGNORE")
        obs = env.step(LeadTriageAction(channel=action))
        if obs.done:
            break

    assert len(obs.contact_history) >= start_len
    assert len(obs.contact_history) <= 10
    assert obs.history_summary.total_touches == len(obs.contact_history)
