"""WebSocket client for the lead triage environment."""

from __future__ import annotations

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import LeadTriageAction, LeadTriageObservation, LeadTriageState


class LeadTriageEnv(EnvClient[LeadTriageAction, LeadTriageObservation, LeadTriageState]):
    """
    Client for LeadTriageEnvironment.

    Example:
        async with LeadTriageEnv(base_url="http://localhost:8000") as env:
            result = await env.reset(seed=42)
            result = await env.step(LeadTriageAction(channel="EMAIL"))
    """

    def _step_payload(self, action: LeadTriageAction) -> Dict[str, Any]:
        return action.model_dump(mode="json")

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[LeadTriageObservation]:
        obs_raw = payload.get("observation", {})
        observation = LeadTriageObservation.model_validate(obs_raw)
        return StepResult(
            observation=observation,
            reward=payload.get("reward", observation.reward),
            done=payload.get("done", observation.done),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> LeadTriageState:
        return LeadTriageState.model_validate(payload)
