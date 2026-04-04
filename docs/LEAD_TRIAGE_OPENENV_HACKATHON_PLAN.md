# Lead Triage OpenEnv — End-to-End Hackathon Plan

A step-by-step plan to build the **sales lead triage** environment (CALL / EMAIL / FOLLOW_UP / IGNORE), meet **OpenEnv** + organizer requirements, and submit.

---

## Phase 0 — Prerequisites (~1 hour)

1. **Accounts & tools:** Hugging Face account, HF token, OpenAI API key (or compatible endpoint), Docker, Python 3.11+, `uv` or `pip` as required by OpenEnv CLI.
2. **Install OpenEnv CLI** (follow current hackathon / OpenEnv docs; exact command may be `pip install openenv` or repo instructions).
3. **Read the official validator doc** once: what `openenv validate` checks, exact `inference.py` contract, required env vars.

---

## Phase 1 — Scaffold the project (~half day)

4. Run `openenv init lead_triage_env` (or your chosen name); open the generated tree.
5. Confirm layout matches expectations: `models.py`, `client.py`, `server/app.py`, `server/environment.py`, `server/Dockerfile`, `openenv.yaml`, `pyproject.toml`.
6. **Pin dependencies** lightly (FastAPI, uvicorn, pydantic, httpx/websockets if needed, `openai`) so **8 GB** builds stay safe.

---

## Phase 2 — Design on paper (~half day)

7. **Observation:** Fixed fields (e.g. `company_size`, `budget`, `industry`, `source`, `engagement_score`, `days_since_contact`) + `step_index` / `max_steps` + optional `legal_actions` / `last_event`.
8. **Actions:** `CALL`, `EMAIL`, `FOLLOW_UP`, `IGNORE` → integers 0–3; document **when FOLLOW_UP is allowed** (mask or penalty).
9. **Episode:** **One lead = one episode**; **max steps** 3–5; **IGNORE** behavior (immediate terminal or one terminal step).
10. **Dynamics:** Transition table: per (tier, action, latent quality) → probabilities for convert / positive reply / no response / churn; **fully specified** in README.
11. **Step rewards:** Conversion, reply, no response, wasted high-cost action, per-step cost, **loop / repeat** penalty.
12. **Three tiers:** Easy / medium / hard = different **noise**, **overlap** of features vs latent quality, **costs**, or **base rates**—same action space.
13. **Grader (0.0–1.0):** One function per tier (or one function + tier param) mapping **trajectory log** → score; **deterministic** given seed + fixed episode set.

---

## Phase 3 — Implement the environment (~1–2 days)

14. **`models.py`:** Pydantic (or whatever `openenv validate` expects) for **Action**, **Observation**, **State**, and **step result** (`reward`, `done`, `info`).
15. **`server/environment.py`:**
    - Internal: lead sample, RNG, step counter, episode log.
    - `reset()` → initial observation (+ pick tier from env var or request).
    - `step(action)` → validate action, sample outcome, update log, return obs/reward/done/info.
    - `state()` → episode metadata (tier, lead id, step count, seeds).
16. **`server/app.py`:** Wire to `create_fastapi_app` (or current OpenEnv pattern); ensure `/health`, `/reset`, `/step`, `/state` match spec.
17. **`client.py`:** `HTTPEnvClient` (or async WebSocket client if template uses it): `_step_payload`, `_parse_result`, `_parse_state`.
18. **Unit-test locally:** `reset` → several `step`s → terminal; illegal actions; max steps; grader on fixed log → expected score.

---

## Phase 4 — `openenv.yaml` & validation (~half day)

19. Fill `openenv.yaml`: name, version, description, any Space-related fields per template.
20. Run `openenv validate`; fix until clean.
21. **Smoke HTTP:** `curl /health`, POST `/reset`, POST `/step` with minimal JSON.

---

## Phase 5 — Baseline `inference.py` (repo root) (~1 day)

22. **`inference.py`:**
    - Load `OPENAI_API_KEY`, `API_BASE_URL`, `MODEL_NAME` (and `HF_TOKEN` if needed for Hub).
    - Use **official OpenAI client** only for LLM calls.
    - For **each tier**: fixed **episode count** and **seeds**; loop `reset` → `step` until `done`.
    - Prompt: map observation JSON → **exactly one** action; `temperature=0`; cap tokens.
    - Print **per-tier grader scores** + summary table; ensure **< 20 min** on reference hardware (trim episodes if needed).
23. **Reproducibility:** Document `SEED` / tier episode list; avoid nondeterministic extras.

---

## Phase 6 — Docker (~half day)

24. **`server/Dockerfile`:** Slim base, install app, `CMD`/`ENTRYPOINT` for uvicorn on `PORT` (often 7860 or 8000 per Space).
25. **Local:** `docker build` + `docker run`; point client at `localhost`; run `inference.py` against local URL first.
26. Fix anything the **automated docker build** would catch (paths, `WORKDIR`, missing files).

---

## Phase 7 — Hugging Face Space (~half day)

27. `openenv push --repo-id your-namespace/lead-triage-env` (or organizer’s required repo name).
28. Set **Space secrets / variables:** e.g. `WORKERS=1` if memory tight, `PORT`, any `LEAD_*` tier defaults if used server-side.
29. Verify: Space loads in browser; `curl https://.../health` → 200; client `reset()` works against **Space URL**.
30. Confirm **install from Space** works if required: `pip install git+https://huggingface.co/spaces/...` and imports match README.

---

## Phase 8 — README & submission polish (~half day)

31. **README:**
    - What the env simulates (stylized CRM).
    - **Observation** / **action** tables.
    - **Reward** table + **grader** formulas (0–1).
    - **Three tasks** (easy / medium / hard) and what changes between them.
    - Local run, Docker run, Space URL, env vars (`API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`, `OPENAI_API_KEY`).
32. **Enumerate tasks** and how validators run graders (if they script it, mirror that in docs).

---

## Phase 9 — Pre-submission checklist (mandatory)

33. Run the **official pre-submission validation script** (organizer-provided); fix all failures.
34. **HF Space:** automated ping → **200** + **reset** OK.
35. **OpenEnv:** validate passes; **step / reset / state** OK.
36. **Docker build** passes on clean clone.
37. **`inference.py`** completes without error; scores in **[0, 1]** for all three tasks.
38. **3+ tasks:** each grader run verified.

---

## Phase 10 — Submit

39. Tag / branch / zip per hackathon instructions; submit **repo link** + **Space URL** + any **secrets** instructions judges need.
40. Keep a **local copy** of exact commit hash you submitted.

---

## Dependency order (what blocks what)

```text
Design (obs/actions/rewards/graders)
    → environment.py + models
    → app + client
    → openenv validate
    → inference.py (against local server)
    → Dockerfile
    → HF push + inference against Space URL
    → README + organizer validator → submit
```

---

## Time estimate (rough)

- **Minimum viable:** ~3–4 focused days if the template is smooth.
- **Buffer:** +1–2 days for validator surprises and Docker/Space quirks.

---

## Organizer requirements (reminder)

| Requirement | Notes |
|-------------|--------|
| Real-world task | Lead triage / CRM simulator (document as stylized). |
| OpenEnv spec | Typed models, `step` / `reset` / `state`, `openenv.yaml`, `openenv validate`. |
| 3 tasks + graders | Easy → hard; scores **0.0–1.0**; deterministic criteria. |
| Reward | Partial progress + penalties (loops, wasted effort). |
| `inference.py` | Repo root; OpenAI client; `OPENAI_API_KEY`, `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` as specified. |
| Infra | **< 20 min** inference; **2 vCPU, 8 GB** target. |
| Deploy | Hugging Face Space + working Dockerfile. |
| README | Env description, spaces, setup. |

---

*Generated for the lead-qualification-agent / OpenEnv hackathon track.*
