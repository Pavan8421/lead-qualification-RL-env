"""Lead triage environment — Phase 2 dynamics (stylized CRM simulator)."""

from __future__ import annotations

import os
import random
import time
import uuid
from typing import Any, List, Literal, Optional, cast

from openenv.core.env_server import Environment

from ..dynamics import LeadOutcome, sample_outcome
from ..features import build_observable_features, sample_latent_quality
from ..grader import grade_episode_from_log
from ..models import LeadEvent, LeadTriageAction, LeadTriageObservation, LeadTriageState
from ..rewards import ignore_opportunity_cost, reward_breakdown, step_reward
from ..task_tier import TierConfig, normalize_tier, TIER_CONFIGS

LeadChannel = Literal["CALL", "EMAIL", "FOLLOW_UP", "IGNORE"]
ALL_ACTIONS: List[LeadChannel] = ["CALL", "EMAIL", "FOLLOW_UP", "IGNORE"]


def _legal_actions(has_prior_contact: bool) -> List[LeadChannel]:
    if has_prior_contact:
        return list(ALL_ACTIONS)
    return ["CALL", "EMAIL", "IGNORE"]


class LeadTriageEnvironment(Environment[LeadTriageAction, LeadTriageObservation, LeadTriageState]):
    """One lead per episode; stochastic transitions; graded trajectory in metadata on terminal steps."""

    SUPPORTS_CONCURRENT_SESSIONS = True
    _illegal_payload_count = 0
    _invalid_follow_up_count = 0
    _consecutive_repeat_count = 0
    _timeout_count = 0
    _episodes_started = 0

    def __init__(self, max_steps: int = 4) -> None:
        super().__init__()
        self._max_steps = max_steps
        self._step_timeout_s = min(
            float(os.environ.get("LEAD_STEP_TIMEOUT_S", "0.5")),
            0.5,
        )
        self._episode_cap = int(os.environ.get("LEAD_EPISODE_CAP", "100000"))
        self._rng = random.Random()
        self._config: TierConfig = TIER_CONFIGS["easy"]
        self._tier_name = "easy"
        self._latent: str = "mid"
        self._features: dict = {}
        self._contact_attempts = 0
        self._has_contact = False
        self._trajectory: List[dict] = []
        self._last_action: Optional[str] = None
        self._action_streak = 0
        self._state: Optional[LeadTriageState] = None
    @classmethod
    def record_illegal_payload(cls) -> None:
        cls._illegal_payload_count += 1

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> LeadTriageObservation:
        if LeadTriageEnvironment._episodes_started >= self._episode_cap:
            raise RuntimeError(
                f"Episode cap reached ({self._episode_cap}); restart server process."
            )
        self._reset_rubric()
        LeadTriageEnvironment._episodes_started += 1
        eid = episode_id or str(uuid.uuid4())
        tier_kw = kwargs.get("task_tier") or os.environ.get("LEAD_TASK_TIER")
        self._tier_name = normalize_tier(str(tier_kw) if tier_kw is not None else None)
        self._config = TIER_CONFIGS[self._tier_name]

        if seed is not None:
            self._rng.seed(seed)
        elif kwargs.get("seed") is not None:
            self._rng.seed(int(kwargs["seed"]))
        else:
            self._rng.seed(self._derive_seed(eid))

        lead_id = str(kwargs.get("lead_id") or f"lead-{eid[:8]}")
        self._latent = sample_latent_quality(self._config, self._rng)
        self._features = build_observable_features(self._latent, self._config, self._rng)
        self._contact_attempts = 0
        self._has_contact = False
        self._trajectory = []
        self._last_action = None
        self._action_streak = 0

        self._state = LeadTriageState(
            episode_id=eid,
            step_count=0,
            task_tier=self._tier_name,
            lead_id=lead_id,
            max_steps=self._max_steps,
            cumulative_reward=0.0,
            converted=False,
            episode_done=False,
        )
        return self._build_observation(
            step_index=0,
            reward=0.0,
            done=False,
            last_event="none",
        )

    def step(
        self,
        action: LeadTriageAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> LeadTriageObservation:
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        if self._state.episode_done:
            return self._build_observation(
                step_index=self._state.step_count,
                reward=0.0,
                done=True,
                last_event="none",
            )

        ch = action.channel
        legal = _legal_actions(self._has_contact)
        if ch not in legal:
            self._state.step_count += 1
            r = -0.4
            LeadTriageEnvironment._invalid_follow_up_count += 1
            self._state.cumulative_reward += r
            streak = self._action_streak + 1 if ch == self._last_action else 1
            self._last_action = ch
            self._action_streak = streak
            self._log_step(ch, "invalid_follow_up", r, dict(invalid=True))
            return self._build_observation(
                step_index=self._state.step_count,
                reward=r,
                done=False,
                last_event="invalid_follow_up",
            )

        streak = self._action_streak + 1 if ch == self._last_action else 1
        self._last_action = ch
        self._action_streak = streak
        repeat_penalty = -0.25 if streak >= 3 else 0.0
        if repeat_penalty < 0.0:
            LeadTriageEnvironment._consecutive_repeat_count += 1

        if ch == "IGNORE":
            base = ignore_opportunity_cost(self._latent)
            sr = step_reward(
                outcome="ignored_terminal",
                action=ch,
                quality=self._latent,
                tier_waste_mult=self._config.waste_penalty_multiplier,
            )
            total_r = base + sr + repeat_penalty
            self._state.step_count += 1
            self._state.cumulative_reward += total_r
            self._state.episode_done = True
            self._log_step(ch, "ignored_terminal", total_r, {})
            return self._finalize_terminal(last_event="ignored_terminal", reward=total_r)

        if ch in ("CALL", "EMAIL", "FOLLOW_UP"):
            self._contact_attempts += 1

        if ch in ("CALL", "EMAIL"):
            self._has_contact = True

        started = time.perf_counter()
        outcome: LeadOutcome | str = sample_outcome(self._latent, ch, self._config, self._rng)
        elapsed = time.perf_counter() - started
        timeout_limit = timeout_s if timeout_s is not None else self._step_timeout_s
        timed_out = elapsed > timeout_limit
        if timed_out:
            LeadTriageEnvironment._timeout_count += 1
            outcome = "horizon"

        breakdown = reward_breakdown(
            outcome=str(outcome),
            action=ch,
            quality=self._latent,
            tier_waste_mult=self._config.waste_penalty_multiplier,
            converted=outcome == "converted",
            step_index=self._state.step_count,
            max_steps=self._max_steps,
            action_parsed=True,
            legal_actions=legal,
            repetition_penalty=repeat_penalty,
            contact_attempts=self._contact_attempts,
            max_contacts=None,
            terminal_grader=0.0,
        )
        total_r = breakdown.total

        self._state.step_count += 1
        self._state.cumulative_reward += total_r

        terminal = outcome == "converted" or outcome == "churned" or timed_out
        if outcome == "converted":
            self._state.converted = True

        if self._state.step_count >= self._max_steps and not terminal:
            terminal = True
            self._state.cumulative_reward -= total_r
            total_r = reward_breakdown(
                outcome="horizon",
                action=ch,
                quality=self._latent,
                tier_waste_mult=self._config.waste_penalty_multiplier,
                converted=False,
                step_index=self._state.step_count,
                max_steps=self._max_steps,
                action_parsed=True,
                legal_actions=legal,
                repetition_penalty=repeat_penalty,
                contact_attempts=self._contact_attempts,
                max_contacts=None,
                terminal_grader=0.0,
            ).total
            self._state.cumulative_reward += total_r
            outcome = "horizon"

        self._log_step(ch, str(outcome), total_r, {})

        if terminal:
            self._state.episode_done = True
            return self._finalize_terminal(
                last_event=cast(LeadEvent, outcome), reward=total_r
            )

        return self._build_observation(
            step_index=self._state.step_count,
            reward=total_r,
            done=False,
            last_event=cast(LeadEvent, outcome),
        )

    @property
    def state(self) -> LeadTriageState:
        if self._state is None:
            return LeadTriageState(
                invalid_follow_up_count=LeadTriageEnvironment._invalid_follow_up_count,
                consecutive_repeat_count=LeadTriageEnvironment._consecutive_repeat_count,
                timeout_count=LeadTriageEnvironment._timeout_count,
                illegal_payload_count=LeadTriageEnvironment._illegal_payload_count,
                episodes_started=LeadTriageEnvironment._episodes_started,
                episode_cap=self._episode_cap,
            )
        return self._state.model_copy(
            update={
                "invalid_follow_up_count": LeadTriageEnvironment._invalid_follow_up_count,
                "consecutive_repeat_count": LeadTriageEnvironment._consecutive_repeat_count,
                "timeout_count": LeadTriageEnvironment._timeout_count,
                "illegal_payload_count": LeadTriageEnvironment._illegal_payload_count,
                "episodes_started": LeadTriageEnvironment._episodes_started,
                "episode_cap": self._episode_cap,
            }
        )

    def _derive_seed(self, episode_id: str) -> int:
        return (hash(episode_id) & 0x7FFFFFFF) or 1

    def _log_step(self, action: str, outcome: str, reward: float, extra: dict) -> None:
        assert self._state is not None
        self._trajectory.append(
            {
                "action": action,
                "outcome": outcome,
                "reward": reward,
                "step_index": self._state.step_count,
                "max_steps": self._max_steps,
                **extra,
            }
        )

    def _build_observation(
        self,
        *,
        step_index: int,
        reward: float,
        done: bool,
        last_event: Any,
        grader_score: Optional[float] = None,
    ) -> LeadTriageObservation:
        return LeadTriageObservation(
            company_size=int(self._features["company_size"]),
            budget=float(self._features["budget"]),
            industry=str(self._features["industry"]),
            source=str(self._features["source"]),
            engagement_score=float(self._features["engagement_score"]),
            days_since_contact=int(self._features["days_since_contact"]),
            intent_score=float(self._features["intent_score"]),
            job_title=str(self._features["job_title"]),
            contact_attempts=int(self._contact_attempts),
            estimated_deal_value=float(self._features["estimated_deal_value"]),
            urgency_level=self._features["urgency_level"],
            step_index=step_index,
            max_steps=self._max_steps,
            has_prior_contact=self._has_contact,
            last_event=last_event,
            legal_actions=_legal_actions(self._has_contact),
            task_tier=self._tier_name,
            grader_score=grader_score,
            trajectory=list(self._trajectory),
            done=done,
            reward=reward,
        )

    def _finalize_terminal(
        self, *, last_event: LeadEvent, reward: float
    ) -> LeadTriageObservation:
        score = grade_episode_from_log(self._trajectory, self._tier_name)
        return self._build_observation(
            step_index=self._state.step_count if self._state else 0,
            reward=reward,
            done=True,
            last_event=last_event,
            grader_score=score,
        )
