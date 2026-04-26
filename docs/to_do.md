Tier 1 — Do these. Highest ROI, low cost.
1. Action with arguments (4 actions → ~10–14)
Today the policy is essentially a 4-way classifier, which is the worst shape for an LLM — there's almost nothing for token generation to learn beyond a label.

Change action to a small structured object:

EMAIL(template ∈ {generic, value_prop, case_study, re_engage})
CALL(script ∈ {discovery, demo, closing})
FOLLOW_UP(channel ∈ {email, call}, tone ∈ {soft, firm})
IGNORE
Effect on dynamics: outcome probabilities now depend on (latent_quality, action, argument) — e.g. case_study lifts conversion on high-intent leads, hurts on low-intent. This creates real text-conditional advantage for the LLM.

Files touched: models.py, dynamics.py, rewards.py. Keep old action shape accepted for backward compat by mapping bare EMAIL → EMAIL(generic).

2. Partial observability
Currently intent_score is given directly — almost a label leak on easy. Restrict so:

intent_score only appears after at least one CALL or EMAIL.
Before first contact, replace with intent_estimate ∈ {low, medium, high, unknown} derived noisily from engagement_score + source.
Why: forces the agent to decide to gather information. Creates an exploration vs exploitation trade-off, which is the entire point of RL. Single biggest lever for making training non-trivial.

Files touched: features.py, lead_triage_environment.py.

3. Cooldowns / temporal spacing
FOLLOW_UP is currently legal immediately after CALL/EMAIL. Add:

Minimum 1 step (= 1 simulated day) between contacts.
Burn-out penalty: >2 contacts within episode reduces conversion probability.
Why: removes the trivial "spam contacts" exploit and makes IGNORE and timing strategic. Anti-reward-hack (guide §8).

Files touched: legality logic in lead_triage_environment.py, small change to legal_actions mask.

4. Verifier panel (multi-reward, independent)
The guide is explicit (§7): use multiple independent reward functions. Today there's effectively one composite scalar.

Split into independent verifiers, each emitted to the trainer as a column:

outcome_reward (existing)
efficiency_reward: bonus for converting in fewer steps
format_compliance: action parses cleanly + obeys legal mask (already implicit, make it explicit)
repetition_penalty (existing — surface separately)
budget_reward: stays under per-episode contact cap
terminal_grader (existing)
Why: lets you watch §15 monitoring properly and detect reward hacking early. Trivial to add now, painful to retrofit.

Files touched: rewards.py returns a RewardBreakdown dataclass; sum stays compatible.

# Action-with-arguments — what it means and why it matters

## The problem with the current setup

Right now the policy outputs **one of 4 strings**: `CALL`, `EMAIL`, `FOLLOW_UP`, `IGNORE`.

That's a classification problem. An LLM has billions of parameters trained to generate fluent text — but here you're using it like a tiny logistic regression. The "intelligence" of the model has nowhere to go. A 50-line scikit-learn classifier on the observation features would do nearly as well, and that's *bad for the demo*: judges will (correctly) say "you didn't need an LLM for this."

## What "action with arguments" means

Instead of a flat label, the action becomes a **structured choice**: a verb + one or more modifiers.

```
EMAIL(template = generic | value_prop | case_study | re_engage)
CALL(script   = discovery | demo | closing)
FOLLOW_UP(channel = email | call, tone = soft | firm)
IGNORE
```

Counting them out:

| Verb | Argument combos | Total |
|---|---|---|
| EMAIL | 4 templates | 4 |
| CALL | 3 scripts | 3 |
| FOLLOW_UP | 2 channels × 2 tones | 4 |
| IGNORE | — | 1 |
| **Total** | | **12** |

So the action space goes from **4 → 12**. Still small enough to enumerate, large enough that *which template you pick* genuinely matters.

## Why arguments make outcomes depend on context

Today the dynamics table is:
```
P(outcome | latent_quality, action)
```
With arguments it becomes:
```
P(outcome | latent_quality, action, argument)
```

Concretely, you'd design the dynamics so different arguments are right for different leads. Example shape (numbers illustrative):

| Lead profile | Best action | Why |
|---|---|---|
| High intent, late-stage (urgency=high, attempts≥1) | `CALL(closing)` | Ready to buy — push to close |
| High intent, cold (no prior contact) | `EMAIL(value_prop)` | Warm them up before a call |
| Mid intent, technical buyer (job_title="VP Eng") | `EMAIL(case_study)` | Proof points convert engineers |
| Low intent | `EMAIL(generic)` or `IGNORE` | Don't burn high-touch budget |
| Stalled lead (days_since_contact > 14) | `EMAIL(re_engage)` | Standard re-engagement play |
| Warm reply received | `FOLLOW_UP(call, firm)` | Strike while hot |
| Polite no-response | `FOLLOW_UP(email, soft)` | Don't burn the relationship |

The wrong argument on the right verb should *hurt*. E.g. `CALL(closing)` on a cold low-intent lead → high `churned` probability. `EMAIL(case_study)` to a non-technical retail buyer → `no_response`.

That's the whole point: **the argument carries information that only matters if you actually read the observation**. A classifier that ignores `job_title` and `intent_score` can't pick the right template. An LLM that reads them naturally can.

## Why this specifically helps an LLM policy

Three reasons:

**1. Token generation is now meaningful.** The model emits something like
```json
{"action":"EMAIL","template":"case_study"}
```
Each token after the first carries decision content. With 4 flat actions, only the first token matters; everything else is decorative. With arguments, a 6-token output is 6 tokens of *policy*, and GRPO's per-token credit assignment has something real to work with.

**2. Reasoning has somewhere to go.** You can prompt the model with a thinking pattern:
```
Lead: VP Eng at fintech, intent=0.78, no prior contact
Reasoning: technical senior buyer, high intent but cold → lead with proof, not a call
Action: EMAIL(case_study)
```
With 4 flat actions, "reasoning" is overkill. With 12 contextual actions, reasoning measurably improves performance — which means RL has something to *learn to do better*.

**3. Headroom over the rule baseline grows.** Your current rule policy in inference.py handles 4 actions with maybe 8 if/else branches. Encoding good 12-action policy in if/else is painful — the rule baseline gets weaker, and the trained LLM has room to clearly outperform it. That's the §19 demo story: "the rule policy plateaus at X, the trained LLM hits Y because it picks template/script per lead."

## What changes in code (sketch, not implementing now)

- models.py: `LeadTriageAction` becomes
  ```
  channel: Literal["CALL","EMAIL","FOLLOW_UP","IGNORE"]
  argument: Optional[str]   # template/script/channel-tone, validated per channel
  ```
  Backward compatible: bare `EMAIL` (no argument) auto-maps to `EMAIL(generic)`.
- dynamics.py: outcome weights become a 3-key lookup `(quality, channel, argument) → weights`. Easiest implementation: per-channel base weights × per-argument multiplier × per-(quality, argument) bonus.
- rewards.py: unchanged, but waste-penalty logic can additionally penalize obviously-wrong arguments (e.g. `CALL(closing)` on no-prior-contact lead).
- `legal_actions` in the observation expands from 4 strings to a list of `(channel, argument)` tuples (or a dict per channel listing legal arguments). Trainer's action mask consumes this directly.
- inference.py: rule policy adds argument selection (~20 lines). LLM prompt asks for the structured form.

## What to watch out for

- **Don't explode the space.** 12 is sweet. Going to 30+ (every combination of every modifier) makes early RL fail because the success probability per random action drops below the trainability threshold (guide §1).
- **Don't make arguments cosmetic.** If outcome probabilities don't actually depend on the argument, the model will learn to ignore it and you've added complexity for nothing. The dynamics table must reward the right argument-for-context.
- **Keep parsing strict.** Reject malformed action JSON at the env boundary (422 in FastAPI). Don't auto-fix — that becomes a reward hack vector.
- **Anchor the grader.** When you change action space, the `oracle_return` and `random_return` anchors in task_tier.py will shift. Recompute them by running an oracle policy (knows latent quality) and a uniform-random policy across many seeds, then update.

## TL;DR

Going from 4 flat actions to ~12 structured actions:
- Doesn't really expand the policy space (still tiny by RL standards).
- Massively expands the **information per decision** — the model must read the observation to pick the right argument.
- Gives the LLM a real reason to exist in this pipeline.
- Strengthens the demo because the rule baseline can't keep up.

It's the single change that converts this env from "a classification benchmark in RL clothing" into "a problem an LLM is genuinely the right tool for."