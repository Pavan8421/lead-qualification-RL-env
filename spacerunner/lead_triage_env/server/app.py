"""FastAPI + WebSocket app for Lead Triage Environment."""

from __future__ import annotations

import os

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from openenv.core.env_server import create_app
from fastapi.exception_handlers import request_validation_exception_handler
from starlette.responses import HTMLResponse

from ..models import LeadTriageAction, LeadTriageObservation
from .lead_triage_environment import LeadTriageEnvironment

app = create_app(
    LeadTriageEnvironment,
    LeadTriageAction,
    LeadTriageObservation,
    env_name="lead_triage_env",
    max_concurrent_envs=int(os.environ.get("LEAD_MAX_CONCURRENT_ENVS", "64")),
)

# Replace default /state so we can surface extended counters.
app.router.routes = [
    route
    for route in app.router.routes
    if not (getattr(route, "path", None) == "/state" and "GET" in getattr(route, "methods", set()))
]


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
):
    if request.url.path.endswith("/step"):
        LeadTriageEnvironment.record_illegal_payload()
    return await request_validation_exception_handler(request, exc)


@app.get("/state")
async def state() -> dict:
    env = LeadTriageEnvironment()
    return env.state.model_dump()


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
