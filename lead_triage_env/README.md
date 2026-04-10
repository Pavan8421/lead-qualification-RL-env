---
title: lead_triage_env
sdk: docker
app_port: 8000
colorFrom: blue
colorTo: green
---

# Lead Qualification RL Environment (OpenEnv Hackathon)

This project implements a real-world inspired **sales lead triage** environment for RL/agent evaluation using OpenEnv.

The environment simulates what sales teams do in CRMs: decide whether to **CALL**, **EMAIL**, **FOLLOW_UP**, or **IGNORE** a lead under uncertainty, cost, and limited outreach budget.

## Problem statement we solve

Many outreach systems optimize one local metric (open rate, response rate) but ignore end-to-end outcomes like conversion, churn risk, and wasted human effort.

This environment models that full decision loop:

- The agent sees noisy lead signals (not latent true quality)
- It makes sequential outreach decisions
- Outcomes are stochastic (convert/reply/no response/churn)
- Rewards include both progress and penalties
- A trajectory-level grader converts performance to a normalized task score

## Task setup

- **Tasks / tiers:** `easy`, `medium`, `hard`
- **Episode unit:** one lead per episode
- **Horizon:** `max_steps = 4`
- **Action space:** `CALL`, `EMAIL`, `FOLLOW_UP`, `IGNORE`
- **Constraint:** `FOLLOW_UP` only legal after prior contact (`CALL` or `EMAIL`)

Tier difficulty is controlled by:

- latent quality priors
- observation noise
- outcome luck and overlap
- waste-penalty multipliers

## Observation space

The agent receives typed observations with both business and control signals:

- **Lead profile:** `company_size`, `budget`, `industry`, `source`, `engagement_score`, `days_since_contact`
- **Realism extensions:** `intent_score`, `job_title`, `contact_attempts`, `estimated_deal_value`, `urgency_level`
- **Control state:** `step_index`, `max_steps`, `has_prior_contact`, `legal_actions`, `last_event`, `task_tier`
- **Terminal scoring context:** `grader_score` (terminal), `trajectory`

## Reward function

Rewards are dense and shaped (see `lead_triage_env/rewards.py`):

- Positive for meaningful progress:
  - `converted`: high positive
  - `positive_reply`: moderate positive
- Negative for poor outcomes:
  - `no_response`, `churned`, `horizon`
- Includes per-step cost
- Penalizes wasted high-touch actions on low-quality leads
- Penalizes bad behavior (loops/repeats, invalid follow-up attempts)
- `IGNORE` adds opportunity-cost logic by latent quality

This gives useful learning signal over the whole trajectory, not just binary terminal success.

## Grader logic (task score in strict (0,1))

The grader (`lead_triage_env/grader.py`) builds a trajectory summary and outputs a normalized score:

- Uses tier-specific anchors (`oracle` vs `random`) for return normalization
- Blends return quality with heuristics:
  - conversion bonus
  - churn penalty
  - repetition penalty
  - early-ignore penalty
  - horizon-without-convert penalty
- Applies strict open-interval clamping so score is always **`0 < score < 1`**

## Baseline inference policy

`inference.py` (repo root) is the baseline runner used for validation:

- Uses **OpenAI client** for all LLM calls
- Reads required env vars (`API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`/API key)
- Runs deterministic seeded episodes across all three tiers
- Emits strict structured logs:
  - `[START]`
  - `[STEP]`
  - `[END] success=... steps=... score=... rewards=...`

The policy is hybrid:

- LLM proposes an action
- Rule guardrails enforce domain-safe decisions using `intent_score`, `contact_attempts`, `legal_actions`, etc.

## Repository layout

- `lead_triage_env/` — environment package (models, dynamics, features, rewards, grader, server, Dockerfile)
- `inference.py` — baseline inference script (must stay at repo root for validator)
- `docs/LEAD_TRIAGE_OPENENV_HACKATHON_PLAN.md` — implementation roadmap
- `docs/PHASE0_PREREQUISITES.md` — setup and prerequisites
- `.env.example` — local environment variable template

## Environment variables

Copy `.env.example` to `.env` (never commit secrets):

- `API_BASE_URL` (OpenAI default: `https://api.openai.com/v1`)
- `MODEL_NAME`
- `HF_TOKEN` or provider API key
- Optional: `LEAD_TRIAGE_ENV_BASE_URL`, `EPISODES_PER_TIER`, `BASE_SEED`, `LOCAL_IMAGE_NAME`

## Run locally

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
EPISODES_PER_TIER=1 python inference.py
```

## Docker

```bash
cd lead_triage_env
docker build -f Dockerfile -t lead-triage-env:local .
docker run --rm -p 8000:8000 --name lead-triage-env-local lead-triage-env:local
```

Then run inference from repo root.

## Hugging Face Space

- Space repo: `pavanKumar2004/lead_triage_env`
- Space URL: `https://pavanKumar2004-lead-triage-env.hf.space`
- Health endpoint: `https://pavanKumar2004-lead-triage-env.hf.space/health`

## Pre-submit checklist

- `openenv validate` passes
- Docker build + run pass
- HF Space is live (`/health`, `/reset`)
- Baseline inference runs successfully and logs strict format
- Task scores remain strictly within `(0,1)`
