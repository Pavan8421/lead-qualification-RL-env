"""Tests for the M5 rollout/training scaffolding (CPU-only, no env required)."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import pytest

from lead_triage_env.training.prompt import (
    build_prompt,
    render_history_narrative,
)
from lead_triage_env.training.rewards import (
    episode_scalar_reward,
    group_advantages,
    summarize_breakdown,
)
from lead_triage_env.training.rollout import (
    action_payload_from_token,
    parse_action_token,
)


# --- prompt -----------------------------------------------------------------


def _fake_observation() -> Dict[str, Any]:
    return {
        "company_size": 250,
        "budget": 50000.0,
        "industry": "tech",
        "source": "linkedin",
        "engagement_score": 0.6,
        "days_since_contact": 2,
        "intent_score": 0.0,
        "intent_estimate": "medium",
        "job_title": "VP Eng",
        "contact_attempts": 1,
        "estimated_deal_value": 25000.0,
        "urgency_level": "medium",
        "step_index": 1,
        "max_steps": 4,
        "has_prior_contact": True,
        "last_event": "no_response",
        "task_tier": "easy",
        "legal_actions": ["CALL", "EMAIL", "FOLLOW_UP", "IGNORE"],
        "legal_action_map": {
            "CALL": ["discovery", "demo"],
            "EMAIL": ["generic", "value_prop"],
            "FOLLOW_UP": ["email:soft"],
            "IGNORE": [""],
        },
        "history_summary": {
            "total_touches": 3,
            "days_since_first_touch": 14,
            "days_since_last_touch": 2,
            "inbound_count": 1,
            "last_inbound_sentiment": 0.4,
            "longest_silence_gap": 7,
            "raised_objections": ["pricing"],
        },
        "contact_history": [
            {
                "day_offset": -14,
                "channel": "EMAIL",
                "direction": "outbound",
                "outcome": "no_response",
                "topic": "intro",
                "sentiment": 0.0,
                "duration_min": 0,
                "note": "intro email",
            },
            {
                "day_offset": -7,
                "channel": "EMAIL",
                "direction": "inbound",
                "outcome": "positive_reply",
                "topic": "pricing question",
                "sentiment": 0.4,
                "duration_min": 0,
                "note": "asked about pricing",
            },
        ],
    }


def test_build_prompt_includes_legal_tokens_and_narrative() -> None:
    bundle = build_prompt(_fake_observation())
    assert bundle["legal_tokens"] == [
        "CALL(discovery)",
        "CALL(demo)",
        "EMAIL(generic)",
        "EMAIL(value_prop)",
        "FOLLOW_UP(email:soft)",
        "IGNORE",
    ]
    user_msg = bundle["messages"][-1]["content"]
    assert "Lead snapshot" in user_msg
    assert "Contact history" in user_msg
    assert "pricing question" in user_msg
    assert "FOLLOW_UP(email:soft)" in user_msg


def test_render_history_with_no_records() -> None:
    txt = render_history_narrative([], None)
    assert "No prior interactions" in txt


# --- parse / payload --------------------------------------------------------


@pytest.mark.parametrize(
    "completion,expected",
    [
        ("EMAIL(value_prop)", "EMAIL(value_prop)"),
        ("the answer is FOLLOW_UP(email:soft).", "FOLLOW_UP(email:soft)"),
        ("CALL", "CALL(discovery)"),
        ("ignore please", "IGNORE"),
        ("nonsense", None),
    ],
)
def test_parse_action_token(completion: str, expected: Any) -> None:
    legal = [
        "CALL(discovery)",
        "EMAIL(value_prop)",
        "FOLLOW_UP(email:soft)",
        "IGNORE",
    ]
    assert parse_action_token(completion, legal) == expected


def test_action_payload_round_trip() -> None:
    assert action_payload_from_token("EMAIL(case_study)") == {
        "channel": "EMAIL",
        "template": "case_study",
    }
    assert action_payload_from_token("CALL(closing)") == {
        "channel": "CALL",
        "script": "closing",
    }
    assert action_payload_from_token("FOLLOW_UP(call:firm)") == {
        "channel": "FOLLOW_UP",
        "follow_up_channel": "call",
        "follow_up_tone": "firm",
    }
    assert action_payload_from_token("IGNORE") == {"channel": "IGNORE"}


# --- rewards ----------------------------------------------------------------


def test_episode_scalar_reward_combines_step_and_grader() -> None:
    s = episode_scalar_reward([1.0, -0.5, 2.0], 0.7, lambda_terminal=2.0)
    assert s.total_step_reward == 2.5
    assert s.terminal_grader == 0.7
    assert s.value == pytest.approx(2.5 + 2.0 * 0.7)


def test_group_advantages_zero_mean_unit_std() -> None:
    rewards = [1.0, 2.0, 3.0, 4.0]
    adv = group_advantages(rewards)
    assert sum(adv) == pytest.approx(0.0, abs=1e-6)
    # std should be ~1
    var = sum(a * a for a in adv) / len(adv)
    assert var == pytest.approx(1.0, rel=1e-3)


def test_group_advantages_singleton() -> None:
    assert group_advantages([3.14]) == [0.0]
    assert group_advantages([]) == []


def test_summarize_breakdown_averages_columns() -> None:
    rows: List[Dict[str, float]] = [
        {"format_compliance": 1.0, "outcome_reward": -1.0},
        {"format_compliance": 0.0, "outcome_reward": 3.0},
    ]
    summary = summarize_breakdown(rows)
    assert summary["format_compliance"] == pytest.approx(0.5)
    assert summary["outcome_reward"] == pytest.approx(1.0)


# --- stub policy + collect_episode (no real env) ----------------------------


class _FakeStepResult:
    def __init__(self, observation: Any, reward: float, done: bool) -> None:
        self.observation = observation
        self.reward = reward
        self.done = done


class _FakeObs:
    """Minimal stand-in for LeadTriageObservation that supports model_dump."""

    def __init__(self, *, legal_tokens: List[str], step_index: int, done: bool,
                 grader: float = 0.0, last_event: str = "none") -> None:
        self._payload = {
            "legal_actions": [t.split("(")[0] for t in legal_tokens],
            "legal_action_map": {
                "EMAIL": ["generic"],
                "IGNORE": [""],
            },
            "history_summary": {},
            "contact_history": [],
            "step_index": step_index,
            "max_steps": 2,
            "has_prior_contact": False,
            "intent_estimate": "medium",
            "intent_score": 0.0,
            "engagement_score": 0.5,
            "task_tier": "easy",
            "last_event": last_event,
        }
        self.last_event = last_event
        self.grader_score = grader
        self.trajectory: List[Dict[str, Any]] = []

    def model_dump(self, mode: str = "json") -> Dict[str, Any]:  # noqa: D401
        return dict(self._payload)


class _FakeEnv:
    def __init__(self) -> None:
        self.calls: List[Any] = []

    async def reset(self, *, seed: int, task_tier: str) -> _FakeStepResult:
        return _FakeStepResult(
            _FakeObs(legal_tokens=["EMAIL(generic)", "IGNORE"], step_index=0, done=False),
            reward=0.0,
            done=False,
        )

    async def step(self, action: Any) -> _FakeStepResult:
        self.calls.append(action)
        # one step + terminal
        terminal = len(self.calls) >= 1
        return _FakeStepResult(
            _FakeObs(
                legal_tokens=["EMAIL(generic)", "IGNORE"],
                step_index=len(self.calls),
                done=terminal,
                grader=0.6,
                last_event="converted" if terminal else "no_response",
            ),
            reward=1.5,
            done=terminal,
        )


def test_collect_episode_with_stub_policy() -> None:
    from lead_triage_env.training.policy import make_stub_policy_fn
    from lead_triage_env.training.rollout import collect_episode

    env = _FakeEnv()
    policy = make_stub_policy_fn()
    rollout = asyncio.run(collect_episode(env, policy, seed=1, tier="easy"))  # type: ignore[arg-type]
    assert rollout.error is None
    assert rollout.num_steps == 1
    assert rollout.terminal_grader == 0.6
    assert rollout.converted is True
    assert rollout.steps[0].action_token == "EMAIL(generic)"
