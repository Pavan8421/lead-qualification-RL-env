"""Lead triage environment — Phase 2 dynamics (stylized CRM simulator)."""

from __future__ import annotations

import os
import random
import time
import uuid
from typing import Any, Dict, List, Literal, Optional, cast

from openenv.core.env_server import Environment

from ..dynamics import LeadOutcome, sample_outcome
from ..features import (
    build_observable_features,
    bucket_intent_score,
    derive_intent_estimate,
    sample_latent_quality,
)
from ..grader import grade_episode_from_log
from ..models import (
    HistorySummary,
    Interaction,
    LeadEvent,
    LeadTriageAction,
    LeadTriageObservation,
    LeadTriageState,
)
from ..persona import PersonaName, sample_history, sample_persona, summarize_history
from ..rewards import ignore_opportunity_cost, reward_breakdown, step_reward
from ..task_tier import TierConfig, normalize_tier, TIER_CONFIGS

LeadChannel = Literal["CALL", "EMAIL", "FOLLOW_UP", "IGNORE"]
ALL_ACTIONS: List[LeadChannel] = ["CALL", "EMAIL", "FOLLOW_UP", "IGNORE"]
EMAIL_TEMPLATES: List[str] = ["generic", "value_prop", "case_study", "re_engage"]
CALL_SCRIPTS: List[str] = ["discovery", "demo", "closing"]
FOLLOW_UP_VARIANTS: List[str] = ["email:soft", "email:firm", "call:soft", "call:firm"]


def _legal_actions_v1(has_prior_contact: bool) -> Dict[str, List[str]]:
    if has_prior_contact:
        return {
            "CALL": list(CALL_SCRIPTS),
            "EMAIL": list(EMAIL_TEMPLATES),
            "FOLLOW_UP": list(FOLLOW_UP_VARIANTS),
            "IGNORE": [""],
        }
    return {
        "CALL": list(CALL_SCRIPTS),
        "EMAIL": list(EMAIL_TEMPLATES),
        "IGNORE": [""],
    }


def _legal_actions_v2(
    *,
    has_prior_contact: bool,
    days_since_contact: int,
    contact_attempts: int,
    max_contacts: int,
) -> Dict[str, List[str]]:
    """v2 legal-action mask:
    - per-episode contact budget (max_contacts) caps CALL/EMAIL/FOLLOW_UP.
    - FOLLOW_UP additionally requires prior contact AND days_since_contact >= 1.
    """
    if contact_attempts >= max_contacts:
        return {"IGNORE": [""]}
    actions: Dict[str, List[str]] = {
        "CALL": list(CALL_SCRIPTS),
        "EMAIL": list(EMAIL_TEMPLATES),
    }
    if has_prior_contact and days_since_contact >= 1:
        actions["FOLLOW_UP"] = list(FOLLOW_UP_VARIANTS)
    actions["IGNORE"] = [""]
    return actions


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
        self._env_version = os.environ.get("LEAD_ENV_VERSION", "v1").strip().lower() or "v1"
        self._max_contacts = max(1, int(os.environ.get("LEAD_MAX_CONTACTS", "3")))
        self._rng = random.Random()
        self._config: TierConfig = TIER_CONFIGS["easy"]
        self._tier_name = "easy"
        self._latent: str = "mid"
        self._features: dict = {}
        self._intent_estimate: str = "unknown"
        self._persona_name: Optional[PersonaName] = None
        self._contact_history: List[Interaction] = []
        self._history_summary: HistorySummary = HistorySummary()
        self._contact_attempts = 0
        self._has_contact = False
        self._trajectory: List[dict] = []
        self._last_action: Optional[str] = None
        self._action_streak = 0
        self._state: Optional[LeadTriageState] = None

    @classmethod
    def record_illegal_payload(cls) -> None:
        cls._illegal_payload_count += 1

    def _current_legal_action_map(self) -> Dict[str, List[str]]:
        if self._env_version == "v2":
            return _legal_actions_v2(
                has_prior_contact=self._has_contact,
                days_since_contact=int(self._features.get("days_since_contact", 0)),
                contact_attempts=self._contact_attempts,
                max_contacts=self._max_contacts,
            )
        return _legal_actions_v1(self._has_contact)

    def _current_legal_actions(self) -> List[LeadChannel]:
        return [cast(LeadChannel, key) for key in self._current_legal_action_map().keys()]

    @staticmethod
    def _action_argument_key(action: LeadTriageAction) -> str:
        if action.channel == "EMAIL":
            return str(action.template or "generic")
        if action.channel == "CALL":
            return str(action.script or "discovery")
        if action.channel == "FOLLOW_UP":
            return f"{action.follow_up_channel or 'email'}:{action.follow_up_tone or 'soft'}"
        return ""

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
        self._intent_estimate = derive_intent_estimate(
            float(self._features["engagement_score"]),
            str(self._features["source"]),
            self._rng,
        )
        self._contact_attempts = 0
        self._has_contact = False
        self._trajectory = []
        self._last_action = None
        self._action_streak = 0
        if self._env_version == "v2":
            self._persona_name = sample_persona(self._latent, self._rng)
            self._contact_history = sample_history(self._persona_name, self._rng)
            self._history_summary = summarize_history(
                self._contact_history, persona=self._persona_name
            )
        else:
            self._persona_name = None
            self._contact_history = []
            self._history_summary = HistorySummary()

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
        legal_map = self._current_legal_action_map()
        legal = [cast(LeadChannel, key) for key in legal_map.keys()]
        arg_key = self._action_argument_key(action)
        is_legal = ch in legal_map and arg_key in legal_map.get(ch, [])
        if not is_legal:
            self._state.step_count += 1
            r = -0.4
            LeadTriageEnvironment._invalid_follow_up_count += 1
            self._state.cumulative_reward += r
            streak = self._action_streak + 1 if ch == self._last_action else 1
            self._last_action = ch
            self._action_streak = streak
            self._log_step(ch, "invalid_follow_up", r, dict(invalid=True, argument=arg_key))
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
            if self._env_version == "v2":
                self._append_agent_interaction(action=ch, argument="", outcome="ignored_terminal")
            self._log_step(ch, "ignored_terminal", total_r, {})
            return self._finalize_terminal(last_event="ignored_terminal", reward=total_r)

        if ch in ("CALL", "EMAIL", "FOLLOW_UP"):
            self._contact_attempts += 1

        if ch in ("CALL", "EMAIL"):
            self._has_contact = True

        industry = (
            str(self._features.get("industry", "")) if self._env_version == "v2" else None
        )
        started = time.perf_counter()
        outcome: LeadOutcome | str = sample_outcome(
            self._latent,
            ch,
            self._config,
            self._rng,
            industry=industry,
            persona=self._persona_name if self._env_version == "v2" else None,
            argument=arg_key,
        )
        elapsed = time.perf_counter() - started
        timeout_limit = timeout_s if timeout_s is not None else self._step_timeout_s
        timed_out = elapsed > timeout_limit
        if timed_out:
            LeadTriageEnvironment._timeout_count += 1
            outcome = "horizon"

        breakdown_max_contacts = self._max_contacts if self._env_version == "v2" else None
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
            max_contacts=breakdown_max_contacts,
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
                max_contacts=breakdown_max_contacts,
                terminal_grader=0.0,
            ).total
            self._state.cumulative_reward += total_r
            outcome = "horizon"

        if self._env_version == "v2" and not terminal:
            if ch in ("CALL", "EMAIL", "FOLLOW_UP"):
                self._features["days_since_contact"] = 0
            self._features["days_since_contact"] = (
                int(self._features.get("days_since_contact", 0)) + 1
            )

        if self._env_version == "v2":
            self._append_agent_interaction(
                action=ch,
                argument=arg_key,
                outcome=cast(LeadEvent, outcome),
            )

        self._log_step(ch, str(outcome), total_r, {"argument": arg_key})

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
        real_intent = float(self._features["intent_score"])
        if self._env_version == "v2":
            if self._has_contact:
                exposed_intent = real_intent
                exposed_estimate = bucket_intent_score(real_intent)
            else:
                exposed_intent = 0.0
                exposed_estimate = self._intent_estimate
        else:
            exposed_intent = real_intent
            exposed_estimate = "unknown"

        if self._env_version == "v2":
            contact_history = list(self._contact_history)
            history_summary = self._history_summary
        else:
            contact_history = []
            history_summary = HistorySummary()
        legal_action_map = self._current_legal_action_map()
        legal_actions = [cast(LeadChannel, key) for key in legal_action_map.keys()]

        return LeadTriageObservation(
            company_size=int(self._features["company_size"]),
            budget=float(self._features["budget"]),
            industry=str(self._features["industry"]),
            source=str(self._features["source"]),
            engagement_score=float(self._features["engagement_score"]),
            days_since_contact=int(self._features["days_since_contact"]),
            intent_score=exposed_intent,
            intent_estimate=exposed_estimate,
            job_title=str(self._features["job_title"]),
            contact_attempts=int(self._contact_attempts),
            estimated_deal_value=float(self._features["estimated_deal_value"]),
            urgency_level=self._features["urgency_level"],
            step_index=step_index,
            max_steps=self._max_steps,
            has_prior_contact=self._has_contact,
            last_event=last_event,
            legal_actions=legal_actions,
            legal_action_map=legal_action_map,
            task_tier=self._tier_name,
            grader_score=grader_score,
            trajectory=list(self._trajectory),
            contact_history=contact_history,
            history_summary=history_summary,
            done=done,
            reward=reward,
        )

    def _append_agent_interaction(
        self,
        *,
        action: LeadChannel,
        argument: str,
        outcome: LeadEvent,
    ) -> None:
        topic = {
            "CALL": "agent call touch",
            "EMAIL": "agent email touch",
            "FOLLOW_UP": "agent follow up",
            "IGNORE": "agent ignore decision",
        }[action]
        sentiment = 0.6 if outcome in ("positive_reply", "converted") else -0.25
        interaction = Interaction(
            day_offset=0,
            channel=action,
            direction="outbound",
            outcome=outcome,
            topic=f"{topic}:{argument}" if argument else topic,
            sentiment=sentiment,
            duration_min=10 if action == "CALL" else 4,
            note=f"Agent action={action} argument={argument or 'none'} produced {outcome}",
        )
        self._contact_history = (self._contact_history + [interaction])[-10:]
        self._history_summary = summarize_history(
            self._contact_history,
            persona=self._persona_name,
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
