"""Typed Action, Observation, and State for the lead triage environment."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server import Action, Observation, State
from pydantic import ConfigDict, Field, model_validator

LeadChannel = Literal["CALL", "EMAIL", "FOLLOW_UP", "IGNORE"]
UrgencyLevel = Literal["low", "medium", "high"]
IntentEstimate = Literal["low", "medium", "high", "unknown"]
InteractionDirection = Literal["outbound", "inbound"]
EmailTemplate = Literal["generic", "value_prop", "case_study", "re_engage"]
CallScript = Literal["discovery", "demo", "closing"]
FollowUpChannel = Literal["email", "call"]
FollowUpTone = Literal["soft", "firm"]
LeadEvent = Literal[
    "none",
    "converted",
    "positive_reply",
    "no_response",
    "churned",
    "ignored_terminal",
    "invalid_follow_up",
    "horizon",
]


class LeadTriageAction(Action):
    """One outreach decision for the current lead."""

    model_config = ConfigDict(extra="forbid", strict=True)

    channel: LeadChannel = Field(
        ...,
        description="Outbound action: CALL, EMAIL, FOLLOW_UP, or IGNORE",
    )
    template: Optional[EmailTemplate] = Field(
        default=None,
        description="EMAIL argument; defaults to generic when omitted",
    )
    script: Optional[CallScript] = Field(
        default=None,
        description="CALL argument; defaults to discovery when omitted",
    )
    follow_up_channel: Optional[FollowUpChannel] = Field(
        default=None,
        description="FOLLOW_UP argument: email or call",
    )
    follow_up_tone: Optional[FollowUpTone] = Field(
        default=None,
        description="FOLLOW_UP argument: soft or firm",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_and_validate(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        channel = data.get("channel")
        if channel == "EMAIL":
            data["template"] = data.get("template") or "generic"
            if any(data.get(k) is not None for k in ("script", "follow_up_channel", "follow_up_tone")):
                raise ValueError("EMAIL action must only include template argument")
            return data
        if channel == "CALL":
            data["script"] = data.get("script") or "discovery"
            if any(data.get(k) is not None for k in ("template", "follow_up_channel", "follow_up_tone")):
                raise ValueError("CALL action must only include script argument")
            return data
        if channel == "FOLLOW_UP":
            data["follow_up_channel"] = data.get("follow_up_channel") or "email"
            data["follow_up_tone"] = data.get("follow_up_tone") or "soft"
            if any(data.get(k) is not None for k in ("template", "script")):
                raise ValueError("FOLLOW_UP action must only include follow_up_channel/follow_up_tone")
            return data
        if channel == "IGNORE":
            if any(data.get(k) is not None for k in ("template", "script", "follow_up_channel", "follow_up_tone")):
                raise ValueError("IGNORE action cannot include arguments")
        return data


class Interaction(Observation):
    day_offset: int = Field(default=0, description="Relative day offset from current step")
    channel: LeadChannel = Field(default="EMAIL")
    direction: InteractionDirection = Field(default="outbound")
    outcome: LeadEvent = Field(default="none")
    topic: str = Field(default="", description="Short topic for this interaction")
    sentiment: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Sentiment for inbound/outbound interaction note",
    )
    duration_min: int = Field(default=0, ge=0)
    note: str = Field(default="", description="Narrative note for prompt rendering")


class HistorySummary(Observation):
    total_touches: int = Field(default=0, ge=0)
    days_since_first_touch: int = Field(default=0, ge=0)
    days_since_last_touch: int = Field(default=0, ge=0)
    inbound_count: int = Field(default=0, ge=0)
    last_inbound_sentiment: float = Field(default=0.0, ge=-1.0, le=1.0)
    longest_silence_gap: int = Field(default=0, ge=0)
    raised_objections: List[str] = Field(default_factory=list)


class LeadTriageObservation(Observation):
    """What the agent sees: lead features plus routing hints (Phase 2 fields)."""

    company_size: int = Field(default=0, ge=0, description="Approximate headcount")
    budget: float = Field(default=0.0, ge=0.0, description="Stated budget (USD)")
    industry: str = Field(default="", description="Industry label")
    source: str = Field(default="", description="Lead source, e.g. linkedin, webform")
    engagement_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Normalized engagement"
    )
    days_since_contact: int = Field(
        default=0, ge=0, description="Days since last outbound touch"
    )
    intent_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Model-estimated purchase intent (noisy vs latent quality). "
            "Masked to 0.0 pre-contact when env_version=v2."
        ),
    )
    intent_estimate: IntentEstimate = Field(
        default="unknown",
        description=(
            "Coarse pre-contact intent bucket derived from engagement_score + source. "
            "Always 'unknown' on env_version=v1 for backward compatibility."
        ),
    )
    job_title: str = Field(default="", description="Contact job title (sampled category)")
    contact_attempts: int = Field(
        default=0,
        ge=0,
        description="Count of outbound CALL/EMAIL/FOLLOW_UP this episode",
    )
    estimated_deal_value: float = Field(
        default=0.0,
        ge=0.0,
        description="Estimated deal size (USD), correlated with budget/quality",
    )
    urgency_level: UrgencyLevel = Field(
        default="medium",
        description="Stated urgency (noisy; harder tiers more misleading)",
    )
    step_index: int = Field(default=0, ge=0, description="Current step index in episode")
    max_steps: int = Field(default=4, ge=1, description="Horizon cap for this episode")
    has_prior_contact: bool = Field(
        default=False,
        description="True after at least one CALL or EMAIL — unlocks FOLLOW_UP",
    )
    last_event: LeadEvent = Field(
        default="none",
        description="Most recent environment event for shaping / debugging",
    )
    legal_actions: List[LeadChannel] = Field(
        default_factory=lambda: ["CALL", "EMAIL", "FOLLOW_UP", "IGNORE"],
        description="Allowed actions this step (FOLLOW_UP gated until contact)",
    )
    legal_action_map: Dict[str, List[str]] = Field(
        default_factory=dict,
        description=(
            "Structured legal actions for M4: channel -> allowed argument keys. "
            "Argument keys are script/template variants or 'channel:tone' for FOLLOW_UP."
        ),
    )
    task_tier: str = Field(
        default="easy",
        description="Current task tier for this episode",
    )
    grader_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Terminal episode score in [0,1]; None for non-terminal steps",
    )
    trajectory: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Episode trajectory records (for debugging/grading fallback)",
    )
    contact_history: List[Interaction] = Field(
        default_factory=list,
        description="Prior interactions for this lead (v2 realism feature)",
    )
    history_summary: HistorySummary = Field(
        default_factory=HistorySummary,
        description="Compact scalar summary of contact history",
    )


class LeadTriageState(State):
    """Episode-level metadata exposed via state()."""

    task_tier: str = Field(
        default="easy",
        description="easy | medium | hard",
    )
    lead_id: str = Field(default="", description="Synthetic lead identifier")
    max_steps: int = Field(default=4, ge=1, description="Episode horizon")
    cumulative_reward: float = Field(
        default=0.0, description="Sum of step rewards so far (server bookkeeping)"
    )
    converted: bool = Field(default=False, description="Whether lead converted this episode")
    episode_done: bool = Field(default=False, description="Terminal flag mirror")
    invalid_follow_up_count: int = Field(
        default=0,
        ge=0,
        description="Count of illegal FOLLOW_UP attempts across this process",
    )
    consecutive_repeat_count: int = Field(
        default=0,
        ge=0,
        description="Count of repeat-streak penalties triggered across this process",
    )
    timeout_count: int = Field(
        default=0,
        ge=0,
        description="Count of per-step timeout events across this process",
    )
    illegal_payload_count: int = Field(
        default=0,
        ge=0,
        description="Count of malformed /step payloads rejected with HTTP 422",
    )
    episodes_started: int = Field(
        default=0,
        ge=0,
        description="Total episodes started by this process",
    )
    episode_cap: int = Field(
        default=100000,
        ge=1,
        description="Maximum episodes allowed for this process",
    )
