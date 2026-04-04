# Phase 0 — Prerequisites (completed checklist)

Use this before running the server, validators, or `inference.py` (later phases).

## Accounts and tooling

| Item | Purpose |
|------|---------|
| **Python 3.10+** (3.11 recommended) | Runtime for env + client |
| **Docker Desktop** (optional Phase 6+) | Container build/run |
| **Hugging Face account + token** | `HF_TOKEN` — Spaces deploy, Hub API |
| **OpenAI-compatible API key** | `OPENAI_API_KEY` — baseline `inference.py` |

## Install OpenEnv CLI and core

From the [OpenEnv README](https://github.com/meta-pytorch/OpenEnv):

```bash
pip install "openenv-core[core]>=0.2.2"
```

This should install the **`openenv`** CLI entry point. If `openenv` is not on your `PATH`, try:

```bash
python -m pip show openenv-core
# Windows: Scripts folder must be on PATH, e.g. Python311\Scripts
```

Useful commands once available:

```bash
openenv --help
openenv init --help
openenv validate   # run before hackathon submit (exact flags per organizer)
```

**Docs:** [OpenEnv documentation](https://meta-pytorch.org/OpenEnv/) — environment builder, deployment, validation.

## Hackathon environment variables (for later phases)

Repo templates: **`.env.example`** (safe to commit) and **`.env`** (gitignored — copy the example and fill in secrets).

| Variable | Role |
|----------|------|
| `OPENAI_API_KEY` | LLM calls in `inference.py` |
| `API_BASE_URL` | Base URL for the OpenAI-compatible API |
| `MODEL_NAME` | Model id for inference |
| `HF_TOKEN` | Hugging Face Hub / Spaces |

Optional for the server: **`HOST`**, **`PORT`** (defaults `0.0.0.0`, `8000`).

## Optional: UV (faster installs)

If you use `uv`:

```bash
uv pip install -e ./lead_triage_env
uv run --project lead_triage_env server
```

## Verify this repo’s package (Phase 1)

```bash
cd lead_triage_env
pip install -e .
python -c "from lead_triage_env import LeadTriageEnv, LeadTriageAction; print('ok')"
```

Run the stub server:

```bash
uvicorn lead_triage_env.server.app:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

**Note:** Phase 0 does not require a working Space or `inference.py`; those come in later phases.

---

## Phase 1 — Scaffold (implemented in this repo)

The **`lead_triage_env/`** package matches the OpenEnv env layout (`models.py`, `client.py`, `server/`, `openenv.yaml`, `pyproject.toml`, Dockerfile):

| Path | Purpose |
|------|---------|
| `lead_triage_env/models.py` | Pydantic `LeadTriageAction` / `Observation` / `State` |
| `lead_triage_env/client.py` | `LeadTriageEnv` (`EnvClient` + WebSocket) |
| `lead_triage_env/server/lead_triage_environment.py` | Stub `Environment` (`reset` / `step` / `state`) |
| `lead_triage_env/server/app.py` | `create_app(...)` + `main()` for uvicorn |
| `lead_triage_env/server/Dockerfile` | CPU-friendly image (build context = `lead_triage_env/`) |

After `pip install -e ./lead_triage_env`, run `openenv validate` when your CLI is available (hackathon pre-submit).
