"""M4 structured action argument tests."""

from __future__ import annotations

import os
import random

import pytest

from lead_triage_env.dynamics import sample_outcome
from lead_triage_env.models import LeadTriageAction
from lead_triage_env.server.lead_triage_environment import LeadTriageEnvironment
from lead_triage_env.task_tier import TIER_CONFIGS


def test_bare_email_backfills_default_template() -> None:
    action = LeadTriageAction(channel="EMAIL")
    assert action.template == "generic"


def test_invalid_channel_argument_combination_raises() -> None:
    with pytest.raises(Exception):
        LeadTriageAction(channel="EMAIL", script="demo")


def test_v2_observation_exposes_structured_legal_action_map() -> None:
    os.environ["LEAD_ENV_VERSION"] = "v2"
    env = LeadTriageEnvironment()
    obs = env.reset(seed=9, task_tier="easy")
    assert "EMAIL" in obs.legal_action_map
    assert "generic" in obs.legal_action_map["EMAIL"]
    assert "CALL" in obs.legal_action_map
    assert "discovery" in obs.legal_action_map["CALL"]


def test_argument_affects_outcome_distribution() -> None:
    tier = TIER_CONFIGS["easy"]
    rng_a = random.Random(123)
    rng_b = random.Random(123)
    n = 1200
    conv_closing = 0
    conv_discovery = 0
    for _ in range(n):
        if (
            sample_outcome(
                quality="high",
                action="CALL",
                tier=tier,
                rng=rng_a,
                argument="closing",
            )
            == "converted"
        ):
            conv_closing += 1
        if (
            sample_outcome(
                quality="high",
                action="CALL",
                tier=tier,
                rng=rng_b,
                argument="discovery",
            )
            == "converted"
        ):
            conv_discovery += 1
    assert conv_closing > conv_discovery
