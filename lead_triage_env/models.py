"""Typed Action, Observation, and State for the lead triage environment."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server import Action, Observation, State
from pydantic import Field

LeadChannel = Literal["CALL", "EMAIL", "FOLLOW_UP", "IGNORE"]
UrgencyLevel = Literal["low", "medium", "high"]
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

    channel: LeadChannel = Field(
        ...,
        description="Outbound action: CALL, EMAIL, FOLLOW_UP, or IGNORE",
    )


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
        description="Model-estimated purchase intent (noisy vs latent quality)",
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
