# Lead qualification agent (OpenEnv hackathon)

Stylized sales lead triage simulator built with OpenEnv.

- **Actions:** `CALL`, `EMAIL`, `FOLLOW_UP`, `IGNORE`
- **Task tiers:** `easy`, `medium`, `hard`
- **Episode:** one synthetic lead, max steps = 4
- **Outputs:** per-step reward and terminal grader score in `[0,1]`

## Repository layout

- **`lead_triage_env/`** — environment package (models, dynamics, features, grader, server, Dockerfile)
- **`inference.py`** — baseline evaluation runner (OpenAI client + required stdout format)
- **`docs/LEAD_TRIAGE_OPENENV_HACKATHON_PLAN.md`** — end-to-end implementation and submission plan
- **`docs/PHASE0_PREREQUISITES.md`** — install/runtime prerequisites
- **`.env.example`** — env-var template (copy to local `.env`)

## Observation and action summary

### Observation (selected fields)

- Lead profile: `company_size`, `budget`, `industry`, `source`, `engagement_score`, `days_since_contact`
- Added realism fields: `intent_score`, `job_title`, `contact_attempts`, `estimated_deal_value`, `urgency_level`
- Control fields: `step_index`, `max_steps`, `has_prior_contact`, `legal_actions`, `last_event`, `task_tier`

### Action space

- `LeadTriageAction(channel=...)` where `channel ∈ {CALL, EMAIL, FOLLOW_UP, IGNORE}`
- `FOLLOW_UP` is legal only after prior contact (`CALL` or `EMAIL`)

## Reward and grader

- Step rewards come from outcome + shaping penalties (`lead_triage_env/rewards.py`)
- Terminal grade comes from trajectory-level rubric (`lead_triage_env/grader.py`)
- Grader score is normalized to `[0,1]` on terminal observations

## Environment variables

Copy `.env.example` to `.env` (local only, do not commit secrets).

- `API_BASE_URL` — OpenAI-compatible endpoint (OpenAI: `https://api.openai.com/v1`)
- `MODEL_NAME` — model id used by `inference.py`
- `OPENAI_API_KEY` — OpenAI key (or compatible provider key if endpoint differs)
- `HF_TOKEN` — Hugging Face token (for HF tooling/deploy)
- Optional: `LEAD_TRIAGE_ENV_BASE_URL`, `EPISODES_PER_TIER`, `BASE_SEED`

## Local run

```bash
cd lead_triage_env
pip install -e .
openenv validate
server
```

Health: `http://localhost:8000/health`

In another terminal:

```bash
cd ..
python inference.py
```

## Docker run

```bash
cd lead_triage_env
docker build -f Dockerfile -t lead-triage-env:local .
docker run --rm -p 8000:8000 --name lead-triage-env-local lead-triage-env:local
```

Then run:

```bash
cd ..
EPISODES_PER_TIER=1 python inference.py
```

## Hugging Face Space

- Space repo: `pavanKumar2004/lead_triage_env`
- Space URL: `https://pavankumar2004-lead-triage-env.hf.space`
- Health: `https://pavankumar2004-lead-triage-env.hf.space/health`

Set `LEAD_TRIAGE_ENV_BASE_URL` to the Space URL before running `inference.py`.

## Submission checklist (quick)

- `openenv validate` passes
- Local server + inference pass
- Docker health + inference pass
- HF Space health + inference pass
- `inference.py` present at repo root and emits `[START]`, `[STEP]`, `[END]`
