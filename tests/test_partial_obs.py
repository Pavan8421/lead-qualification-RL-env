"""M3a — Partial observability tests."""

from __future__ import annotations

import os

import pytest

from lead_triage_env.features import derive_intent_estimate
from lead_triage_env.models import LeadTriageAction
from lead_triage_env.server.lead_triage_environment import LeadTriageEnvironment


_VALID_ESTIMATES = {"low", "medium", "high", "unknown"}


@pytest.fixture(autouse=True)
def _reset_class_counters():
    LeadTriageEnvironment._invalid_follow_up_count = 0
    LeadTriageEnvironment._consecutive_repeat_count = 0
    LeadTriageEnvironment._timeout_count = 0
    LeadTriageEnvironment._illegal_payload_count = 0
    LeadTriageEnvironment._episodes_started = 0
    yield


def _make_env(version: str) -> LeadTriageEnvironment:
    os.environ["LEAD_ENV_VERSION"] = version
    return LeadTriageEnvironment()


def test_v1_keeps_real_intent_score_and_unknown_estimate() -> None:
    env = _make_env("v1")
    obs = env.reset(seed=1234, task_tier="easy")
    assert obs.intent_score > 0.0
    assert obs.intent_estimate == "unknown"


def test_v2_masks_intent_score_pre_contact() -> None:
    env = _make_env("v2")
    obs = env.reset(seed=1234, task_tier="easy")
    assert obs.intent_score == 0.0
    assert obs.intent_estimate in _VALID_ESTIMATES


def test_v2_unmasks_intent_after_first_contact() -> None:
    env = _make_env("v2")
    env.reset(seed=42, task_tier="easy")
    obs = env.step(LeadTriageAction(channel="EMAIL"))
    assert obs.has_prior_contact is True
    assert obs.intent_score > 0.0
    assert obs.intent_estimate in {"low", "medium", "high"}


def test_derive_intent_estimate_returns_valid_bucket() -> None:
    import random

    rng = random.Random(0)
    for _ in range(50):
        bucket = derive_intent_estimate(rng.random(), "linkedin", rng)
        assert bucket in _VALID_ESTIMATES
