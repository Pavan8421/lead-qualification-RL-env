"""FastAPI + WebSocket app for Lead Triage Environment."""

from __future__ import annotations

import os

from openenv.core.env_server import create_app

from ..models import LeadTriageAction, LeadTriageObservation
from .lead_triage_environment import LeadTriageEnvironment

app = create_app(
    LeadTriageEnvironment,
    LeadTriageAction,
    LeadTriageObservation,
    env_name="lead_triage_env",
)


def main() -> None:
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
