# Task split: 2 people, env vs training

Aligned with [docs/roadmap.md](roadmap.md). Minimizes merge conflicts; both people can work in parallel after a small synchronization point.

---

## Person A — **Env Owner** (server, dynamics, realism)

Owns everything under `lead_triage_env/` **except** the new `lead_triage_env/training/` package. Keeps the OpenEnv validator, Docker build, and HF Space deployable at all times.

| Milestone | Scope | Key files |
|---|---|---|
| **M1** | Per-step wall-clock timeout (≤500 ms), 422 on malformed actions, RNG audit, `/state` counters (`invalid_follow_up`, repeat, timeout, illegal-payload) | [lead_triage_env/server/app.py](../lead_triage_env/server/app.py), [lead_triage_env/server/lead_triage_environment.py](../lead_triage_env/server/lead_triage_environment.py) |
| **M2** | Refactor rewards into `RewardBreakdown` columns: `outcome_reward`, `efficiency_reward`, `format_compliance`, `repetition_penalty`, `budget_compliance`, `terminal_grader`. **Sum stays unchanged** for backward compat | [lead_triage_env/rewards.py](../lead_triage_env/rewards.py), [lead_triage_env/models.py](../lead_triage_env/models.py), `tests/test_rewards.py` |
| **M3a** | Hide `intent_score` until first contact; expose noisy `intent_estimate ∈ {low,medium,high,unknown}` derived from `engagement_score` + `source` | [lead_triage_env/features.py](../lead_triage_env/features.py) |
| **M3b** | `FOLLOW_UP` requires `days_since_contact ≥ 1`; per-episode `max_contacts = 3`; update `legal_actions` mask | [lead_triage_env/server/lead_triage_environment.py](../lead_triage_env/server/lead_triage_environment.py), [lead_triage_env/dynamics.py](../lead_triage_env/dynamics.py) |
| **M3c** | Sample `industry` once per `reset()` (not per step); make outcome probabilities depend on it | [lead_triage_env/dynamics.py](../lead_triage_env/dynamics.py) |
| **M3d** | New `Interaction` + `HistorySummary` Pydantic models; new `persona.py` with 5–6 archetypes (`evaluator_engaged`, `evaluator_stalled`, `champion_internal_blocker`, `tire_kicker`, `ghost`, `re_engaged_after_silence`); wire `reset()` to seed 3–10 prior interactions and `step()` to append. **Never expose `latent_quality` or `persona_name`**. Cap history at ~10 records | [lead_triage_env/models.py](../lead_triage_env/models.py), `lead_triage_env/persona.py` (new), [lead_triage_env/server/lead_triage_environment.py](../lead_triage_env/server/lead_triage_environment.py), `tests/test_persona.py` |
| **M4** | Action-with-arguments (~12 actions: `EMAIL(template)`, `CALL(script)`, `FOLLOW_UP(channel,tone)`, `IGNORE`); persona-aware dynamics `P(outcome \| quality, channel, argument, persona)`; recompute grader oracle/random anchors per tier; update rule policy in `inference.py` for new action shape | [lead_triage_env/models.py](../lead_triage_env/models.py), [lead_triage_env/dynamics.py](../lead_triage_env/dynamics.py), [lead_triage_env/task_tier.py](../lead_triage_env/task_tier.py), [inference.py](../inference.py) |

Person A also bumps an `env_version` flag on each behavior change so v1 keeps passing the OpenEnv validator, and redeploys the HF Space at the **end of M3d** (the natural pause point for §13 deploy-early).

---

## Person B — **Training Owner** (rollouts, GRPO, eval)

Owns the new `lead_triage_env/training/` package and all repo-root training/eval scripts. Treats the env as a black box behind [lead_triage_env/client.py](../lead_triage_env/client.py).

| Milestone | Scope | Key files |
|---|---|---|
| **M0** | Lock model (`Qwen2.5-1.5B-Instruct`, fallback `0.5B`), algo (GRPO), efficiency (Unsloth 4-bit + LoRA), logging (W&B), no-SFT decision, disjoint seed plan (train `[0,10000)`, eval `[9000,9100)` per tier) | `requirements-train.txt` (new), `configs/training.yaml` (new) |
| **M5 prompt** | `build_prompt(obs)` → chat messages: system + structured observation + **rendered narrative `contact_history`** (M3d) + structured `legal_actions`. Mirror `inference.py`'s message shape for fair before/after | `lead_triage_env/training/prompt.py` |
| **M5 rollout** | `async collect_episode(env, policy_fn, seed, tier)` + `collect_batch(...)`; **resample on illegal action** (do NOT eat the −0.4 penalty); capture per-step rewards, breakdown columns, terminal grader, invalid/repeat counts | `lead_triage_env/training/rollout.py`, `lead_triage_env/training/policy.py` |
| **M5 rewards** | `episode_scalar = Σ per_step_rewards + λ * terminal_grader` (default λ=2.0); GRPO group standardization (G samples / prompt); emit M2 breakdown dict to W&B columns | `lead_triage_env/training/rewards.py`, `tests/test_rewards.py` |
| **M6** | TRL `GRPOTrainer` with `reward_funcs=[...]` from M2/M5; Unsloth `FastLanguageModel.from_pretrained(load_in_4bit=True)`; LoRA via PEFT (rank 16, α 32, target=all linear); KL `beta=0.04`; `num_generations=4`; save adapters every N steps to `outputs/adapters/` | `lead_triage_env/training/trainer_grpo.py`, `train.py`, `configs/training.yaml` |
| **M7** | 50–200 step smoke run on `easy`; W&B dashboard for total/per-column reward, KL, action entropy across all 12 actions, invalid-resample rate; **hard-fail loud** on >2% invalid, >85% single-action, grader ≤ rule baseline, KL>10; `inspect_rollouts.py` dumps 20 sample episodes | `scripts/inspect_rollouts.py` |
| **M8** | Curriculum sampler: `easy` only → 50/50 with `medium` when easy grader > 0.55 → 1/3 each when medium > 0.50; rolling-buffer driven | `lead_triage_env/training/rollout.py` (sampler), `configs/training.yaml` |
| **M9** | K parallel env containers via `asyncio.Semaphore(K)`; bump G 4 → 8; optional vLLM-via-Unsloth for 3–5× rollout speedup (§12) | `lead_triage_env/training/rollout.py`, infra scripts |
| **M10** | Save LoRA adapters (no naive 4-bit→fp16 merge, §16); `eval.py --policy {rule,llm,trained}` reusing `[START]/[STEP]/[END]` log format on fixed seed grid; `compare.py` table per tier including per-component rewards; README update; HF Space demo with `contact_history` rendering | `eval.py`, `scripts/compare.py`, `scripts/merge_to_fp16.py` (optional), `README.md` |

---

## Shared / synchronization points

These are the only places both people touch — coordinate via short-lived branches.

1. **`models.py` schema (M2 + M3d + M4)** — Person A owns it. Person B must approve any field rename/removal. **Lock the observation schema at the end of M3d** so Person B can build M5 prompt rendering against a frozen contract. Action shape is re-locked at the end of M4.
2. **Reward breakdown column names (M2)** — Person A defines column names in `RewardBreakdown`; Person B consumes them in `training/rewards.py`. Agree on the exact column list in a 10-min sync **before** M2 lands. (Names from the roadmap: `outcome_reward`, `efficiency_reward`, `format_compliance`, `repetition_penalty`, `budget_compliance`, `terminal_grader`.)
3. **Action space change (M4)** — Breaking. Person A ships M4 on a feature branch; Person B updates `prompt.py` legal-action rendering + `rollout.py` legal-mask resample logic on the same branch **before merge to main**.
4. **`inference.py` rule policy** — Person A owns updates for new history/action shape; Person B uses `inference.py` as the `--policy rule` baseline in `eval.py` / `compare.py`. Both must keep its `[START]/[STEP]/[END]` log format intact (the OpenEnv validator depends on it).
5. **Env `version` flag** — Person A bumps it on each behavior change (M3a, M3b, M3d, M4). Person B's `train.py` and `eval.py` should log the env version they ran against, so W&B runs are comparable.

---

## Suggested timeline (dependency-correct, parallel where possible)

```
A:  M1 → M2 ──┬─ M3a → M3b → M3c → M3d ──┬─ M4 ─────────────────────────► (env stable)
              │                            │
              │ (sync: reward cols)        │ (sync: schema freeze)         (sync: action space)
              ▼                            ▼                               ▼
B:  M0 ──── (wait on M2 cols) ── M5 prompt+rollout+rewards skeleton ── M5 final → M6 → M7 ─► M8/M9 → M10
```

Person B is **not blocked** during A's M1/M3a–c: they can land M0, scaffold the `training/` package with empty modules, and write a `train.py` stub that just instantiates the env client and prints one rollout.

The hard gates for B are:
- **M2 reward columns** (needed for `training/rewards.py`)
- **M3d schema freeze** (needed to render `contact_history` as narrative)
- **M4 action space** (needed to mask the new 12-action space and parse structured completions)

Two natural deploy/checkpoint moments to align on (mirroring [docs/roadmap.md](roadmap.md)):
1. **End of M3d** — Person A redeploys HF Space; both run a baseline pass to confirm rule-policy grader is sensible.
2. **End of M7** — first trained checkpoint; team decides M8/M9 scale vs jump to M10 demo.

---

## First concrete actions

- **Person A**: open a branch, ship **M1 + M2** as one PR (env hardening + verifier panel — non-breaking; sum of breakdown columns must equal current `step_reward` for `inference.py` to pass unchanged).
- **Person B**: open a branch, ship **M0 + scaffolding**:
  - `requirements-train.txt`
  - `configs/training.yaml` (with all M0 decisions filled in)
  - `lead_triage_env/training/` package with empty `prompt.py`, `rollout.py`, `rewards.py`, `policy.py`, `trainer_grpo.py`
  - `train.py` stub at repo root that loads YAML, opens an env client, runs one rollout with a dummy random policy, prints reward + grader.

Both PRs are independent and can land in either order. After they merge, A starts M3a–d while B starts implementing real `prompt.py` and `rollout.py` against the current observation schema (re-rendering after the M3d schema freeze).