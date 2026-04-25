"""M3b — Cooldown + per-episode contact budget tests."""

from __future__ import annotations

import os

import pytest

from lead_triage_env.models import LeadTriageAction
from lead_triage_env.server.lead_triage_environment import (
    LeadTriageEnvironment,
    _legal_actions_v1,
    _legal_actions_v2,
)


@pytest.fixture(autouse=True)
def _reset_class_counters():
    LeadTriageEnvironment._invalid_follow_up_count = 0
    LeadTriageEnvironment._consecutive_repeat_count = 0
    LeadTriageEnvironment._timeout_count = 0
    LeadTriageEnvironment._illegal_payload_count = 0
    LeadTriageEnvironment._episodes_started = 0
    yield


def test_v2_followup_blocked_when_days_since_contact_zero() -> None:
    legal = _legal_actions_v2(
        has_prior_contact=True,
        days_since_contact=0,
        contact_attempts=1,
        max_contacts=3,
    )
    assert "FOLLOW_UP" not in legal
    assert "CALL" in legal and "EMAIL" in legal


def test_v2_followup_legal_after_one_day_cooldown() -> None:
    legal = _legal_actions_v2(
        has_prior_contact=True,
        days_since_contact=1,
        contact_attempts=1,
        max_contacts=3,
    )
    assert "FOLLOW_UP" in legal
    assert "email:soft" in legal["FOLLOW_UP"]


def test_v2_budget_caps_contacts_to_ignore_only() -> None:
    legal = _legal_actions_v2(
        has_prior_contact=True,
        days_since_contact=5,
        contact_attempts=3,
        max_contacts=3,
    )
    assert legal == {"IGNORE": [""]}


def test_v1_legal_actions_unchanged_no_contact() -> None:
    assert set(_legal_actions_v1(False).keys()) == {"CALL", "EMAIL", "IGNORE"}


def test_v1_legal_actions_unchanged_with_contact() -> None:
    assert set(_legal_actions_v1(True).keys()) == {"CALL", "EMAIL", "FOLLOW_UP", "IGNORE"}


def test_v2_runtime_days_increments_after_step() -> None:
    os.environ["LEAD_ENV_VERSION"] = "v2"
    env = LeadTriageEnvironment()
    env.reset(seed=99, task_tier="easy")

    initial_days = int(env._features["days_since_contact"])
    obs = env.step(LeadTriageAction(channel="EMAIL"))

    if not obs.done:
        assert obs.has_prior_contact is True
        assert obs.days_since_contact == 1
    else:
        assert env._features["days_since_contact"] in (0, initial_days)


def test_v2_observation_drops_contact_actions_after_budget() -> None:
    os.environ["LEAD_ENV_VERSION"] = "v2"
    os.environ["LEAD_MAX_CONTACTS"] = "1"
    env = LeadTriageEnvironment()
    obs = env.reset(seed=7, task_tier="easy")
    assert "CALL" in obs.legal_actions
    assert "CALL" in obs.legal_action_map

    obs = env.step(LeadTriageAction(channel="EMAIL"))
    if not obs.done:
        assert "CALL" not in obs.legal_actions
        assert "EMAIL" not in obs.legal_actions
        assert "FOLLOW_UP" not in obs.legal_actions
        assert "IGNORE" in obs.legal_actions
        assert obs.legal_action_map == {"IGNORE": [""]}

    os.environ.pop("LEAD_MAX_CONTACTS", None)
