# Phase 2 design — Lead triage environment

## Episode

- **One lead = one episode.**
- **Horizon:** `max_steps = 4` (configurable on `LeadTriageEnvironment(max_steps=...)`).
- **IGNORE:** immediate terminal; reward uses `ignore_opportunity_cost(latent_quality)` (high-quality leads penalized more for ignoring).
- **FOLLOW_UP:** allowed only after at least one **CALL** or **EMAIL** (`has_prior_contact` / `legal_actions`).

## Task tiers

| Tier   | Engagement noise | Outcome luck | Waste penalty mult. | Prior \(P(\mathrm{low},\mathrm{mid},\mathrm{high})\) |
|--------|------------------|-------------|---------------------|------------------------------------------------------|
| easy   | ±0.06            | 1.15        | 1.0                 | 0.20 / 0.35 / 0.45                                   |
| medium | ±0.14            | 1.00        | 1.25                | 0.35 / 0.35 / 0.30                                   |
| hard   | ±0.28            | 0.88        | 1.55                | 0.45 / 0.35 / 0.20                                   |

Code: `task_tier.py`, `features.sample_latent_quality`, `TIER_CONFIGS`.

## Latent quality

Server-side only: `low` | `mid` | `high`. Drives sampling weights in `dynamics.sample_outcome` and feature centers in `features.build_observable_features`. The agent only sees noisy observables.

### Extra observable fields

- **`intent_score`** \([0,1]\): centered by latent quality with tier engagement noise.
- **`job_title`**: drawn from a quality-specific title pool.
- **`estimated_deal_value`**: USD, scaled from `budget` with quality-dependent multipliers.
- **`urgency_level`**: `low` \| `medium` \| `high`; on **hard** tier, sometimes resampled uniformly to add misleading urgency.
- **`contact_attempts`**: starts at 0; increments once per executed **CALL**, **EMAIL**, or **FOLLOW_UP** (not **IGNORE**; illegal **FOLLOW_UP** does not increment).

## Transitions (stylized)

Each step, given latent quality and action, `sample_outcome` draws among:

- **converted**
- **positive_reply** (non-terminal — nurture continues)
- **no_response**
- **churned** (terminal)

Relative weights depend on \((\mathrm{quality}, \mathrm{action})\); see `_weights_for` in `dynamics.py`. **IGNORE** bypasses sampling → `ignored_terminal`.

If the step budget is reached without convert/churn, the final event is **horizon** (timeout), with reward from `rewards.step_reward(outcome="horizon")`.

## Step rewards (`rewards.py`)

| Outcome / rule | Value (before repeat penalty) |
|----------------|-------------------------------|
| converted | +10 |
| positive_reply | +3 |
| no_response | −1 |
| churned | −4 |
| horizon | −1.5 (+ step cost) |
| Step cost | −0.15 per non-terminal step |
| Wasted CALL on low quality (+ no_response/churned) | extra −2 × waste_mult |
| Wasted EMAIL on low (+ no_response/churned) | extra −0.5 × waste_mult |
| Same action ≥ 3 times in a row | −0.25 |
| Invalid FOLLOW_UP (no prior contact) | −0.4 |

## Grader \([0,1]\)

`grader.grade_episode` / `grade_episode_from_log`:

- Blends **normalized return** (vs tier `grader_oracle_return` / `grader_random_return` anchors) with a **heuristic** (conversion bonus, churn / loop / early-ignore penalties).
- On terminal steps, `metadata["grader_score"]` is attached to the observation.

Deterministic given fixed **seed** and **tier** passed to `reset`.

## Reset parameters

- `task_tier`: `"easy"` | `"medium"` | `"hard"` (or env `LEAD_TASK_TIER`).
- `seed`, `episode_id`, optional `lead_id`.
