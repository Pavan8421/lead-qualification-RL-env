"""FastAPI + WebSocket app for Lead Triage Environment."""

from __future__ import annotations

import os

from openenv.core.env_server import create_app
from starlette.responses import HTMLResponse

from ..models import LeadTriageAction, LeadTriageObservation
from .lead_triage_environment import LeadTriageEnvironment

app = create_app(
    LeadTriageEnvironment,
    LeadTriageAction,
    LeadTriageObservation,
    env_name="lead_triage_env",
)


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return """
    <html>
      <head><title>Lead Triage Env</title></head>
      <body style="font-family: Arial, sans-serif; margin: 2rem;">
        <h1>Lead Triage Environment</h1>
        <p>Service is running.</p>
        <ul>
          <li><a href="/health">/health</a></li>
          <li>POST /reset</li>
          <li>POST /step</li>
          <li>GET /state</li>
        </ul>
      </body>
    </html>
    """


def main() -> None:
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
