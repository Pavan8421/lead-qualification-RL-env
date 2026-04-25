# Complete roadmap — guide-aligned, problem-aligned

Each milestone maps to one or more sections of the hackathon guide (cited as **§N**).
Order is dependency-correct; do not skip ahead.

---

## Todo list (authoritative)

```
[ ] M0  Lock decisions (model/algo/logging/SFT)
[x] M1  Env hardening (per-step timeout, 422, counters)
[x] M2  Verifier panel — multi-reward breakdown
[x] M3a Partial observability (gate intent_score)
[x] M3b Cooldowns + per-episode contact budget
[x] M3c Per-episode industry fix
[x] M3d Lead pre-history at reset (persona-driven, 5–10 prior interactions)
[x] M3d Persona archetypes (5–6) + sampler
[ ] M3d Render history as narrative for prompt builder
[x] M4  Action-with-arguments (4 → ~12 actions)
[x] M4  Recompute grader oracle/random anchors
[x] M5  Rollout collector (async, legal-mask resample)
[x] M5  Reward shaping + GRPO group advantages
[x] M6  GRPO trainer (TRL + Unsloth + LoRA)
[x] M6  train.py + configs/training.yaml + requirements-train.txt
[x] M7  Smoke run on easy + W&B inspection
[x] M7  Anti-hack audit (rates, KL, diversity)
[ ] M8  Curriculum sampler easy → medium → hard
[ ] M9  Scale: parallel envs + larger G + vLLM/Unsloth
[ ] M10 Save LoRA adapters + eval.py + compare.py
[ ] M10 Update README + HF Space demo
[v2] (deferred) Cross-episode persistent lead store (resume nurture)
```

---

## M0 — Lock decisions (§0, §3, §10)

Before any code, fix the choices that bias every later step.

- **Model:** `Qwen2.5-1.5B-Instruct` (laptop fallback `Qwen2.5-0.5B-Instruct`). Small enough to run with Unsloth 4-bit on a single 16–24 GB GPU; instruct-tuned so JSON-style action emission works out of the box.
- **Algorithm:** GRPO via TRL (§11). No value model, group-baseline advantages, ideal for short stochastic episodes.
- **Efficiency layer:** Unsloth 4-bit + LoRA (§10, §16) — patches HF model with faster Triton kernels and one-line PEFT.
- **Logging:** Weights & Biases (§15) — required to monitor per-reward columns, KL, action diversity.
- **SFT first?** No. Start RL on the instruct base. Only revisit SFT warm-start if zero-reward rate stays >70% after M7 (§3).
- **Rollout transport:** keep current OpenEnv async HTTP client; one env server, N async workers via `asyncio.gather`.
- **Seed plan:** disjoint ranges per tier per phase — train on `[0, 10_000)`, eval on `[9_000, 9_100)` per tier; never overlap.

**Exit:** decisions captured in `configs/training.yaml` at M6.

---

## M1 — Env hardening (§4, §8, §14)

Close the §8 anti-hacking gaps **before** training, otherwise a learned policy will exploit them.

Edits in [lead_triage_env/server/lead_triage_environment.py](lead_triage_env/server/lead_triage_environment.py) and [lead_triage_env/server/app.py](lead_triage_env/server/app.py):

- **Per-`step` wall-clock timeout** (≤500 ms; abort → `last_event="horizon"`, terminal). Wraps the dynamics call in `asyncio.wait_for` (or sync equivalent).
- **422 on malformed action payloads** at the FastAPI boundary — never silently coerce. Validates `channel` ∈ legal enum, validates argument shape after M4.
- **Episode/session caps** to bound state growth (e.g. 100k episodes per server process).
- **Counters** for `invalid_follow_up`, consecutive-repeat, timeout, illegal-payload — exposed via `/state` for §15 monitoring.
- **RNG audit** — confirm `_rng.seed` is called exactly once per `reset()` and never re-seeded mid-episode (no cross-episode leakage).

**Exit:**
- `EPISODES_PER_TIER=1 python inference.py` still passes end-to-end.
- `GET /state` returns the new counters.
- Sending a malformed `POST /step` body returns HTTP 422.

---

## M2 — Verifier panel (§7, §9, §15)

The guide is explicit: **multiple independent reward functions**, not one composite. Single-scalar rewards are easy to hack (§8).

Refactor [lead_triage_env/rewards.py](lead_triage_env/rewards.py) so it returns a `RewardBreakdown` dataclass with independent columns:

| Column | Source | Why it's independent |
|---|---|---|
| `outcome_reward` | event-based table (existing) | The "did the right thing happen" signal |
| `efficiency_reward` | `+0.5` if `converted` AND `step_index < max_steps − 1` | Rewards short, decisive successes |
| `format_compliance` | `1.0` if action parsed cleanly + ∈ `legal_actions`, else `0.0` | Pure structural verifier |
| `repetition_penalty` | existing repeat-streak logic, surfaced separately | Process check on degenerate behavior |
| `budget_compliance` | `1.0` if total contacts ≤ `max_contacts` (M3b), else `0.0` | Resource-usage verifier |
| `terminal_grader` | existing `grader_score`, terminal only | Trajectory-level outcome verifier |

The **scalar sum** stays backward compatible with the existing `step_reward(...)` return so [inference.py](inference.py) keeps working. Each column flows separately into TRL's `reward_funcs` list (M6) → W&B logs each as its own scalar (§15).

**Exit:**
- Unit test in `tests/test_rewards.py` confirms columns sum to the legacy total.
- `inference.py` log output unchanged.

---

## M3 — Env content improvements (non-breaking, version-gated)

Four small env upgrades, all gated behind an `env_version` config flag so v1 keeps passing the OpenEnv validator.

### M3a — Partial observability (§1)
Edit [lead_triage_env/features.py](lead_triage_env/features.py):
- Hide `intent_score` until first `CALL`/`EMAIL`.
- Pre-contact, replace with `intent_estimate ∈ {low, medium, high, unknown}` derived noisily from `engagement_score` + `source`.

**Why:** `intent_score` is currently a near-label-leak on `easy`. Hiding it forces explore-vs-exploit, which is the entire point of RL. Single biggest signal-quality lever.

### M3b — Cooldowns + per-episode contact budget (§8)
Edit `step()` legality logic in [lead_triage_env/server/lead_triage_environment.py](lead_triage_env/server/lead_triage_environment.py):
- `FOLLOW_UP` requires `days_since_contact ≥ 1`.
- Per-episode `max_contacts = 3`; further contact actions become illegal and are masked out of `legal_actions`.

**Why:** removes the "spam contacts" exploit. Combines with M4 to force template selection under scarcity.

### M3c — Per-episode industry (correctness fix)
[lead_triage_env/dynamics.py](lead_triage_env/dynamics.py): sample `industry` once per `reset()`, not per step. Outcome probabilities now depend on industry (e.g. `finance` slower-converting, `tech` faster). Small bug fix masquerading as a feature.

### M3d — Lead pre-history at reset (NEW — biggest realism + LLM-signal upgrade)
Today an episode begins with zero history. In reality every lead has weeks/months of prior touches. Generate that synthetically at reset so the agent must reason over a relationship narrative.

**Data model in [lead_triage_env/models.py](lead_triage_env/models.py):**
- `Interaction(day_offset: int, channel: Literal[...], direction: Literal["outbound","inbound"], outcome: LeadEvent, topic: str, sentiment: float, duration_min: int, note: str)`
- `HistorySummary(total_touches, days_since_first_touch, days_since_last_touch, inbound_count, last_inbound_sentiment, longest_silence_gap, raised_objections: List[str])`
- Add to `LeadTriageObservation`:
  - `contact_history: List[Interaction] = []`
  - `history_summary: HistorySummary = HistorySummary()`
- Defaults are empty/zero so existing clients (validator, current `inference.py`) keep working.

**Persona generator in new module `lead_triage_env/persona.py`:**
- 5–6 archetypes: `evaluator_engaged`, `evaluator_stalled`, `champion_internal_blocker`, `tire_kicker`, `ghost`, `re_engaged_after_silence`.
- Each is a small Markov chain `(latent_quality, rng) → (persona_name, List[Interaction])`.
- Sample 3–10 interactions per lead with realistic `day_offset`s spanning a few weeks.
- The persona drives BOTH the visible history AND the (otherwise unchanged) outcome probabilities, so the history is genuinely informative.

**Wiring:**
- `reset()` samples persona → generates `contact_history` + computes `history_summary` → populates observation.
- `step()` appends the agent's just-executed interaction each step; recomputes `history_summary` cheaply.
- The narrative consumer (the LLM prompt builder in M5) renders `contact_history` as readable text.
- The rule policy in [inference.py](inference.py) reads only `history_summary` (scalars), keeping the rule baseline weak so the LLM has clear headroom.

**Anti-hack (§8):**
- **Never** expose `latent_quality` or `persona_name` in the observation — agent must infer from history.
- Cap history at ~10 records to bound prompt size.
- History at `reset()` is immutable during the episode; only the agent's own actions append.

**No reward changes in this milestone.** Outcomes still come from the same `(quality, channel, argument)` table.

**Exit (M3 overall):**
- Baseline mean grader on `easy` drops slightly (env is harder) but stays well above the tier-random anchor.
- `inference.py` rule baseline still runs; new fields render as text in `[STEP]` logs without crashing.

---

## M4 — Action-with-arguments (§1 — make the LLM the right tool)

Single biggest demo lever. Today the policy is a 4-way classifier; expand to ~12 structured actions so token generation actually carries decision content.

### Action shape ([lead_triage_env/models.py](lead_triage_env/models.py))
```
EMAIL(template ∈ {generic, value_prop, case_study, re_engage})
CALL(script   ∈ {discovery, demo, closing})
FOLLOW_UP(channel ∈ {email, call}, tone ∈ {soft, firm})
IGNORE
```
Counts: 4 + 3 + 4 + 1 = **12 distinct actions**.

**Backward compat:** bare `EMAIL` (no argument) → `EMAIL(generic)` so `inference.py` and the OpenEnv validator keep passing.

### Dynamics ([lead_triage_env/dynamics.py](lead_triage_env/dynamics.py))
Outcome probability becomes `P(outcome | latent_quality, channel, argument, persona)`. Implementation: per-channel base weights × per-argument multiplier × `(quality, argument)` bonus × `(persona, argument)` bonus.

Examples:
- `CALL(closing)` on high-intent → +conversion; on low-intent / cold → +churn.
- `EMAIL(case_study)` on technical title (`VP Eng`) → +positive_reply; else neutral.
- `EMAIL(re_engage)` on `days_since_contact > 14` → +positive_reply; else penalty.
- `FOLLOW_UP(call, firm)` after `inbound replied_positive` → +conversion; on `ghost` persona → +churn.

### Legal-action mask
`legal_actions` becomes a list of `(channel, argument)` tuples (or `{channel: [args]}`). The trainer's mask consumes this directly; the rollout collector resamples on misses (M5).

### Recompute grader anchors ([lead_triage_env/task_tier.py](lead_triage_env/task_tier.py))
Run an oracle policy (knows `latent_quality` + best `(channel, argument)`) and a uniform-random policy across many seeds; update `oracle_return` / `random_return` per tier. Without this, normalized grader scores are wrong after the action-space change.

**Exit:**
- Rule-policy baseline grader degrades vs v1 (good — leaves headroom for RL).
- LLM baseline (current OpenAI call in [inference.py](inference.py)) with a reasoning prompt clearly beats the rule policy.

---

## M5 — Rollout collector + reward shaping (§2, §11)

New package `lead_triage_env/training/`:

### `prompt.py`
- `build_prompt(obs)` returns chat messages: system + structured observation + **rendered narrative `contact_history`** (M3d) + `legal_actions`.
- Mirrors [inference.py](inference.py)'s message shape so before/after comparisons are fair.

### `rollout.py`
- `async collect_episode(env, policy_fn, seed, tier) -> EpisodeRollout`
  - On each step: serialize obs → prompt → sample completion → parse action → **resample** if action ∉ `legal_actions` (do NOT let the trainer absorb the −0.4 invalid penalty as a reward-hack vector).
  - Captures: prompts, responses, per-step rewards, breakdown columns, terminal grader, tier, seed, invalid-resample count, repeat count, full trajectory.
- `async collect_batch(env, policy_fn, n_per_tier, tiers)` — fan-out via `asyncio.gather` against multiple env workers.
- Returns `EpisodeRollout` dataclass.

### `rewards.py`
- `episode_scalar = sum(per_step_rewards) + λ * terminal_grader` (default λ=2.0; `configs/training.yaml`).
- **GRPO grouping:** for each prompt, GRPO samples G completions → standardize advantages within the group of G.
- Emit breakdown dict (the M2 columns) → W&B columns (§15).

**Exit:** `python -m lead_triage_env.training.rollout --tier easy --n 4` against a running env produces 4 well-formed episodes with non-None grader and clean W&B logs.

---

## M6 — GRPO trainer scaffolding (§10, §11, §12, §16)

New files:

- **`requirements-train.txt`**: `trl>=0.11`, `transformers>=4.45`, `accelerate`, `peft`, `bitsandbytes`, `unsloth`, `wandb`, `pyyaml`.

- **`lead_triage_env/training/policy.py`** — Unsloth `FastLanguageModel.from_pretrained(load_in_4bit=True)`; greedy decode (8–32 tokens after M4); regex-extract structured action; **mask to `legal_actions`** before submission to env.

- **`lead_triage_env/training/trainer_grpo.py`** — wraps `trl.GRPOTrainer`:
  - `reward_funcs` is a list of M2 verifiers; each calls `rollout.collect_episode(...)` once per generation and returns one scalar (or column).
  - LoRA via PEFT (rank 16, alpha 32, target = all linear).
  - KL `beta` configurable for §8 drift control (`beta=0.04` default).
  - `num_generations=G` (start 4, raise to 8 in M9).

- **`train.py`** (repo root) — loads YAML, starts env client, instantiates trainer, runs `.train()`, saves LoRA adapters every N steps to `outputs/adapters/` (§16: never naively merge from 4-bit).

- **`configs/training.yaml`** — model, lr, G, episodes/epoch, λ, KL beta, tier mix, seed ranges, eval cadence.

**Exit:**
- `python train.py --config configs/training.yaml --steps 1` instantiates trainer against a running env and completes one optimizer step.
- W&B run shows reward + breakdown columns + KL.

---

## M7 — Train small + inspect (§14 phase 5–6, §15, §8)

Phase 5/6 of the guide. **Look at outputs, not just metrics.**

1. 50–200 step run on `easy` only.
2. W&B watches simultaneously:
   - mean total reward, mean grader,
   - **each M2 breakdown column**,
   - invalid-resample rate, repeat rate,
   - action-distribution entropy across all 12 actions,
   - KL to base model.
3. `scripts/inspect_rollouts.py` dumps 20 sampled episodes (full prompts, responses, per-step rewards, history, trajectory).
4. **Hard fail loud** if any of:
   - `invalid_follow_up` rate > 2%
   - any single action chosen > 85% on `easy` (collapse)
   - mean grader on `easy` ≤ rule baseline from [inference.py](inference.py)
   - KL to base > 10 (drift / hacking risk)
5. If failures: tune λ down, raise KL `beta`, shrink learning rate, or add a format-only warmup phase (reward only `format_compliance` for first N steps).

**Exit:** trained policy strictly beats rule baseline on `easy` mean grader, with healthy action diversity (entropy > 1.5 nats over 12 actions).

---

## M8 — Curriculum (§6, §14 phase 7)

Once `easy` is solved cleanly, expand task diversity gradually so success probability never drops to zero (§1).

- Epoch 0–N₁: 100% `easy`.
- When rolling mean grader on `easy` > 0.55: mix 50/50 `easy`/`medium`.
- When rolling mean grader on `medium` > 0.50: 1/3 each tier.
- Implement as a `tier_sampler` callback in `train.py` driven by a rolling buffer of recent grader scores per tier.

**Exit:** all three tiers train without grader collapse on prior tiers (no catastrophic forgetting).

---

## M9 — Scale (§12, §14 phase 8)

Only after M7/M8 are green.

- Spin up K env containers (`docker run -p 800K:8000 lead-triage-env:local`); collector load-balances via `asyncio.Semaphore(K)`.
- Bump `num_generations` (G) per prompt 4 → 8 (better GRPO baseline).
- Increase episodes/epoch.
- If GPU is the bottleneck: switch generation backend to **vLLM-via-Unsloth** for 3–5× rollout speedup (§12).

**Exit:** episodes/min ≥ 2× M7 throughput; grader keeps trending up; no deadlocks under parallel rollouts.

---

## M10 — Save, eval, demo (§16, §19, §18 phase 9)

The §19 demo story: baseline → verifier output → trained → measurable improvement → safeguards.

- **Save adapters only** with `model.save_pretrained()`. Optional `scripts/merge_to_fp16.py` reloads base in fp16 and merges — **never** dequantize 4-bit then merge (§16).
- **`eval.py`** at repo root, reusing [inference.py](inference.py)'s `[START]/[STEP]/[END]` log format, with `--policy {rule,llm,trained}` flag. Fixed seed grid (e.g. seeds 9000–9099 per tier) for reproducibility.
- **`scripts/compare.py`** prints a table for `rule` vs baseline `llm` vs `trained` per tier:
  - mean grader, conversion rate, invalid rate, mean steps,
  - **plus per-component reward** (M2 columns).
- **Update [README.md](README.md)** with: training command, eval command, results table, adapter-loading instructions, anti-hack safeguards section (§19 bullet 5), team-split mention (§17, optional).
- **HF Space demo:** push trained adapter to HF Hub; optionally extend the existing Space with a small UI showing baseline vs trained on the same lead, **rendering the `contact_history` for visual punch**.

**Exit:** reproducible measurable uplift on `easy` and `medium` over rule baseline; single-command train (`python train.py`) and single-command eval (`python eval.py --policy trained`).

---

## v2 (deferred) — Cross-episode persistent lead store

Server-side `lead_id → history[]` map; `reset(lead_id=X)` resumes the relationship; supports the "resume nurture" story across many training episodes. **Skip for hackathon** — pre-history at reset (M3d) gives 90% of the benefit at 10% of the complexity. Persistent state breaks parallel rollouts, GRPO grouping, and HF Space hosting assumptions.

---

## File deltas (additive, validator-safe)

```
NEW  requirements-train.txt
NEW  configs/training.yaml
NEW  train.py
NEW  eval.py
NEW  scripts/inspect_rollouts.py
NEW  scripts/compare.py
NEW  scripts/merge_to_fp16.py            (optional)
NEW  lead_triage_env/persona.py                     ← M3d
NEW  lead_triage_env/training/__init__.py
NEW  lead_triage_env/training/prompt.py             ← renders contact_history narrative
NEW  lead_triage_env/training/rollout.py
NEW  lead_triage_env/training/rewards.py
NEW  lead_triage_env/training/policy.py
NEW  lead_triage_env/training/trainer_grpo.py
NEW  tests/test_rewards.py
NEW  tests/test_persona.py                          ← M3d
EDIT lead_triage_env/server/lead_triage_environment.py   (M1, M3a–d)
EDIT lead_triage_env/server/app.py                       (M1: 422)
EDIT lead_triage_env/models.py                           (M2 breakdown, M3d Interaction/HistorySummary, M4 action shape)
EDIT lead_triage_env/dynamics.py                         (M3c, M4 dynamics, M3d persona-aware)
EDIT lead_triage_env/features.py                         (M3a partial obs)
EDIT lead_triage_env/rewards.py                          (M2 breakdown)
EDIT lead_triage_env/task_tier.py                        (M4 anchors recompute)
EDIT inference.py                                        (rule policy reads history_summary; structured action handled)
EDIT README.md                                           (M10)
```

---

## Suggested execution order (dependency-correct)

```
M0 → M1 → M2 → M3a → M3b → M3c → M3d → M4 → M5 → M6 → M7 → M8 → M9 → M10
                                  ▲                  ▲
                                  │                  │
                          deploy v2 env to     first trained
                          HF Space (§13)       checkpoint
```

Two natural pause points:

1. **After M3d** — env has realism, partial obs, cooldowns, budget, persona-history. Push to HF Space, sanity-check rule baseline (§13 deploy early).
2. **After M7** — first trained checkpoint. Decide whether to invest in M8/M9 scale or jump straight to M10 demo.

---

## Mapping to the guide (so judges can verify coverage)

| Guide § | Where it lives in this roadmap |
|---|---|
| §1 task properties (verifiable, non-zero success) | M3a (partial obs), M3d (history), M4 (action richness) |
| §2 minimum RL loop | M5 + M6 |
| §3 SFT vs RL | M0 decision |
| §4 env-first design | already done; M1 hardening |
| §5 OpenEnv build | already done |
| §6 curriculum | M8 |
| §7 multiple rewards | M2 |
| §8 anti reward-hack | M1 (timeouts, 422, counters), M3b (budget), M3d (no latent leak), M5 (resample on illegal), M7 (KL audits) |
| §9 process feedback | M2 efficiency / format columns |
| §10 stack | M0, M6 |
| §11 GRPO / RLVR | M6 |
| §12 fast inference | M9 (vLLM/Unsloth) |
| §13 deploy early | already done; redeploy after M3d |
| §14 stable then scale | M7 → M8 → M9 ordering |
| §15 monitoring | M2 columns + M7 W&B dashboard |
| §16 saving | M10 |
| §17 team split | optional, document in README |
| §18 phases 1–9 | M1 → M10 |
| §19 demo | M10 |
| §21 mistakes | M3a/3b/3d/4 (zero-reward avoidance), M2 (single reward), M7 (inspection) |

---

## Next action

Recommended first patch: **M1 + M2 together** — env hardening + verifier panel, both non-breaking.

After that, **M3d** is the highest-leverage realism upgrade and the one that makes the LLM clearly the right tool for the job.