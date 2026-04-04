# Lead qualification agent (OpenEnv hackathon)

- **`docs/LEAD_TRIAGE_OPENENV_HACKATHON_PLAN.md`** — full end-to-end plan  
- **`docs/PHASE0_PREREQUISITES.md`** — tooling, OpenEnv install, env vars  
- **`lead_triage_env/`** — OpenEnv package (Phase 1 scaffold: models, client, stub server, Dockerfile)
- **`.env`** — local secrets (gitignored); copy from **`.env.example`** and set `OPENAI_API_KEY`, `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` as needed for inference / Hub.

## Quick start

```bash
cd lead_triage_env
pip install -e .
server
```

Health: `http://localhost:8000/health`
