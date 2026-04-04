# Lead Triage Environment (OpenEnv)

**Phase 2:** tiered **easy / medium / hard** tasks, **latent lead quality** (server-only), **stochastic transitions**, **shaped rewards**, **repeat / invalid-action penalties**, and a deterministic **grader** in \([0,1]\) on terminal steps (`metadata["grader_score"]`).

Full spec: [`docs/DESIGN_PHASE2.md`](docs/DESIGN_PHASE2.md).

## Layout

```
lead_triage_env/
├── models.py          # Pydantic Action / Observation / State
├── client.py          # WebSocket EnvClient
├── task_tier.py       # Tier hyperparameters
├── dynamics.py        # Outcome sampling
├── features.py        # Observable features from latent quality
├── rewards.py         # Step rewards
├── grader.py          # episode → [0,1]
├── docs/DESIGN_PHASE2.md
├── server/
│   ├── app.py
│   └── lead_triage_environment.py
└── ...
```

## Install

```bash
cd lead_triage_env
pip install -e .
```

## Run server

```bash
server
# Or
uvicorn lead_triage_env.server.app:app --host 0.0.0.0 --port 8000
```

Optional env: **`LEAD_TASK_TIER`** (`easy` | `medium` | `hard`) as default tier when `reset` does not pass `task_tier`.

## Client (async)

```python
import asyncio
from lead_triage_env import LeadTriageEnv, LeadTriageAction

async def main():
    async with LeadTriageEnv(base_url="http://localhost:8000") as env:
        r = await env.reset(seed=42, task_tier="medium")
        print(r.observation.legal_actions)  # FOLLOW_UP gated until CALL/EMAIL
        r = await env.step(LeadTriageAction(channel="EMAIL"))
        print(r.reward, r.done, r.observation.metadata.get("grader_score"))

asyncio.run(main())
```

## Observation highlights

| Field | Meaning |
|--------|---------|
| `company_size`, `budget`, `industry`, `source`, `engagement_score`, `days_since_contact` | Noisy lead profile (correlated with latent quality) |
| `intent_score` | \([0,1]\) intent signal (noisy vs true quality) |
| `job_title` | Sampled title bucket by quality tier |
| `contact_attempts` | Outbound **CALL** / **EMAIL** / **FOLLOW_UP** count this episode (starts at 0) |
| `estimated_deal_value` | USD deal estimate (scaled from budget / quality) |
| `urgency_level` | `low` \| `medium` \| `high` (hard tier can be misleading) |
| `step_index` / `max_steps` | Progress within episode |
| `has_prior_contact` | `True` after CALL or EMAIL — **FOLLOW_UP** allowed |
| `legal_actions` | Subset of `CALL`, `EMAIL`, `FOLLOW_UP`, `IGNORE` |
| `last_event` | Last stochastic / control event (`converted`, `horizon`, …) |

## State (`state()`)

Includes `task_tier`, `lead_id`, `max_steps`, `cumulative_reward`, `converted`, `episode_done`.

## Action space

`LeadTriageAction(channel=...)` with `channel ∈ {CALL, EMAIL, FOLLOW_UP, IGNORE}`.

See `models.py` for full types.
