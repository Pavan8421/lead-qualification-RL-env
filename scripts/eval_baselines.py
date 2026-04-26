"""Baseline evaluation: how good is the system *without* RL training?

Runs N policies through the env using the SAME seeds and reports per-tier
+ overall metrics. Headroom = (max possible score) − (best baseline score).
If headroom is small, RL training won't help much.

Policies evaluated:
  random      — uniform over legal actions
  always_call — picks CALL whenever legal (greedy "high-effort")
  always_email — picks EMAIL whenever legal (greedy "low-effort")
  rule        — the hand-written `_rule_policy_action` from inference.py
  llm:<name>  — any OpenAI-compatible model (e.g. base Qwen, trained adapter)

Prereqs:
  * Env server on http://localhost:8000
  * For LLM policies: an OpenAI-compatible API (e.g. vLLM on :8001)

Examples:
  # Just the cheap baselines (no LLM needed):
  python scripts/eval_baselines.py \
    --policies random always_call always_email rule \
    --episodes-per-tier 16

  # Add base + trained LLM:
  python scripts/eval_baselines.py \
    --policies random rule "llm:Qwen/Qwen2.5-1.5B-Instruct" llm:trained \
    --api-base-url http://127.0.0.1:8001/v1 \
    --episodes-per-tier 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

# ---- WS keepalive patch (CPU-vLLM friendly) --------------------------------
import openenv.core.env_client as _oec  # noqa: E402

_orig_ws_connect = _oec.ws_connect
def _patched_ws_connect(*args, **kwargs):
    kwargs.setdefault("ping_interval", None)
    kwargs.setdefault("ping_timeout", None)
    return _orig_ws_connect(*args, **kwargs)
_oec.ws_connect = _patched_ws_connect
# ----------------------------------------------------------------------------

from lead_triage_env import LeadTriageAction, LeadTriageEnv  # noqa: E402

# Reuse helpers from inference.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inference import (  # noqa: E402
    ALL_ACTIONS,
    DEFAULT_ARG_BY_CHANNEL,
    EPS,
    TIERS,
    _build_action_payload,
    _clamp_open01,
    _extract_grader_score,
    _flatten_legal_tokens,
    _rule_policy_action,
    _safe_float,
    choose_action as llm_choose_action,
)


# ---- policies --------------------------------------------------------------

PolicyFn = Callable[[Dict[str, Any], List[str], Optional[Dict[str, List[str]]]],
                    Tuple[Dict[str, Any], str]]


def _payload_for(channel: str, observation: Dict[str, Any], legal_map: Optional[Dict[str, List[str]]]) -> Tuple[Dict[str, Any], str]:
    arg = DEFAULT_ARG_BY_CHANNEL.get(channel, "")
    if legal_map and channel in legal_map and legal_map[channel]:
        if arg not in legal_map[channel]:
            arg = legal_map[channel][0]
    token = f"{channel}({arg})" if channel != "IGNORE" else "IGNORE"
    return _build_action_payload(channel, arg), token


def policy_random(rng: random.Random) -> PolicyFn:
    def fn(obs, legal, legal_map):
        channel = rng.choice(legal)
        return _payload_for(channel, obs, legal_map)
    return fn


def policy_always(channel: str) -> PolicyFn:
    def fn(obs, legal, legal_map):
        chosen = channel if channel in legal else legal[0]
        return _payload_for(chosen, obs, legal_map)
    return fn


def policy_rule() -> PolicyFn:
    def fn(obs, legal, legal_map):
        channel = _rule_policy_action(obs, legal)
        return _payload_for(channel, obs, legal_map)
    return fn


def policy_llm(client, model_name: str) -> PolicyFn:
    def fn(obs, legal, legal_map):
        payload, token, _err = llm_choose_action(client, model_name, obs, legal, legal_action_map=legal_map)
        return payload, token
    return fn


# ---- episode runner --------------------------------------------------------

@dataclass
class EpisodeResult:
    policy: str
    tier: str
    seed: int
    success: bool = False
    steps: int = 0
    score: float = 0.0
    rewards: List[float] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)


async def run_one(env: LeadTriageEnv, policy: PolicyFn, policy_name: str,
                  tier: str, seed: int, blocking: bool) -> EpisodeResult:
    reset = await env.reset(seed=seed, task_tier=tier)
    obs = reset.observation
    res = EpisodeResult(policy=policy_name, tier=tier, seed=seed)
    done = bool(reset.done)
    score = EPS
    while not done:
        obs_dict = obs.model_dump(mode="json")
        legal = list(obs.legal_actions or ALL_ACTIONS)
        legal_map = getattr(obs, "legal_action_map", None)
        legal_map = legal_map if isinstance(legal_map, dict) else None

        if blocking:
            payload, token = await asyncio.to_thread(policy, obs_dict, legal, legal_map)
        else:
            payload, token = policy(obs_dict, legal, legal_map)

        step = await env.step(LeadTriageAction(**payload))
        obs = step.observation
        done = bool(step.done)
        reward = _safe_float(step.reward)
        res.actions.append(token)
        res.rewards.append(reward)
        res.steps += 1
        if done:
            score = _extract_grader_score(
                getattr(obs, "grader_score", None),
                getattr(obs, "trajectory", None),
                getattr(obs, "metadata", None),
                tier,
            )
            score = _clamp_open01(score)
    res.success = True
    res.score = score
    return res


# ---- aggregation -----------------------------------------------------------

def _agg(eps: List[EpisodeResult]) -> Dict[str, float]:
    if not eps:
        return {"n": 0, "score_mean": 0.0, "score_std": 0.0, "avg_reward": 0.0, "avg_steps": 0.0}
    scores = [e.score for e in eps]
    rewards = [sum(e.rewards) for e in eps]
    return {
        "n": len(eps),
        "score_mean": statistics.fmean(scores),
        "score_std": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
        "avg_reward": statistics.fmean(rewards),
        "avg_steps": statistics.fmean([e.steps for e in eps]),
    }


def _action_dist(eps: List[EpisodeResult]) -> Dict[str, float]:
    counts: Dict[str, int] = {}
    total = 0
    for e in eps:
        for a in e.actions:
            ch = a.split("(")[0]
            counts[ch] = counts.get(ch, 0) + 1
            total += 1
    return {k: v / total for k, v in sorted(counts.items())} if total else {}


# ---- main ------------------------------------------------------------------

async def _async_main(args: argparse.Namespace) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build policy registry
    policies: Dict[str, Tuple[PolicyFn, bool]] = {}
    rng = random.Random(args.base_seed)
    client = None

    for spec in args.policies:
        if spec == "random":
            policies[spec] = (policy_random(rng), False)
        elif spec == "always_call":
            policies[spec] = (policy_always("CALL"), False)
        elif spec == "always_email":
            policies[spec] = (policy_always("EMAIL"), False)
        elif spec == "always_follow_up":
            policies[spec] = (policy_always("FOLLOW_UP"), False)
        elif spec == "always_ignore":
            policies[spec] = (policy_always("IGNORE"), False)
        elif spec == "rule":
            policies[spec] = (policy_rule(), False)
        elif spec.startswith("llm:"):
            if client is None:
                from openai import OpenAI
                api_key = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY") or "dummy"
                client = OpenAI(base_url=args.api_base_url, api_key=api_key)
            model_name = spec[len("llm:"):]
            policies[spec] = (policy_llm(client, model_name), True)
        else:
            raise SystemExit(f"Unknown policy spec: {spec}")

    # Connect once, reuse for all policies (same seeds across policies = paired comparison).
    env = LeadTriageEnv(base_url=args.env_base_url)
    await env.connect()

    all_results: Dict[str, List[EpisodeResult]] = {p: [] for p in policies}

    try:
        for tier_index, tier in enumerate(TIERS):
            for i in range(args.episodes_per_tier):
                seed = args.base_seed + (tier_index * 10000) + i
                for name, (fn, blocking) in policies.items():
                    print(f"  policy={name:<40s} tier={tier} seed={seed} ...", end="", flush=True)
                    res = await run_one(env, fn, name, tier, seed, blocking=blocking)
                    all_results[name].append(res)
                    print(f"  score={res.score:.3f}  steps={res.steps}", flush=True)
    finally:
        await env.close()

    # ---- report -----------------------------------------------------------
    print("\n" + "=" * 92)
    print("BASELINE EVALUATION — same seeds, paired comparison")
    print("=" * 92)

    # Overall table
    print(f"\n{'policy':<40}{'n':>5}{'score μ':>11}{'score σ':>11}{'reward μ':>11}{'steps μ':>10}")
    print("-" * 92)
    for name in policies:
        a = _agg(all_results[name])
        print(f"{name:<40}{a['n']:>5d}{a['score_mean']:>11.4f}{a['score_std']:>11.4f}"
              f"{a['avg_reward']:>11.3f}{a['avg_steps']:>10.2f}")

    # Per-tier
    print("\n--- per-tier mean grader score ---")
    print(f"{'policy':<40}" + "".join(f"{t:>12}" for t in TIERS))
    print("-" * 92)
    for name in policies:
        eps = all_results[name]
        by_tier = {t: [e for e in eps if e.tier == t] for t in TIERS}
        row = f"{name:<40}"
        for t in TIERS:
            row += f"{_agg(by_tier[t])['score_mean']:>12.4f}"
        print(row)

    # Action distributions
    print("\n--- action distribution (channel) ---")
    channels = ["CALL", "EMAIL", "FOLLOW_UP", "IGNORE"]
    print(f"{'policy':<40}" + "".join(f"{c:>11}" for c in channels))
    print("-" * 92)
    for name in policies:
        d = _action_dist(all_results[name])
        row = f"{name:<40}"
        for c in channels:
            row += f"{d.get(c, 0.0) * 100:>10.1f}%"
        print(row)

    # Headroom analysis
    print("\n--- headroom analysis ---")
    best_baseline = max(
        (n for n in policies if not n.startswith("llm:")),
        key=lambda n: _agg(all_results[n])["score_mean"],
        default=None,
    )
    if best_baseline:
        best_b = _agg(all_results[best_baseline])["score_mean"]
        print(f"Best non-LLM baseline:  {best_baseline:<25s}  score = {best_b:.4f}")
    llm_results = {n: _agg(all_results[n]) for n in policies if n.startswith("llm:")}
    for name, a in llm_results.items():
        delta = a["score_mean"] - (best_b if best_baseline else 0.0)
        verdict = "WORSE than rules!" if delta < -0.01 else (
            "marginal" if abs(delta) < 0.05 else "improved"
        )
        print(f"  {name:<40} score = {a['score_mean']:.4f}   Δ vs best baseline = {delta:+.4f}  [{verdict}]")

    print("\nInterpretation:")
    print("  * If LLM ≤ rule baseline   → RL training is unlikely to help (env is rule-solvable).")
    print("  * If LLM ≫ rule baseline   → there's signal; RL gains compound on top.")
    print("  * If rule ≪ ceiling (1.0)  → there's headroom for RL to exploit.")

    # Persist
    payload = {
        name: [
            {
                "tier": e.tier,
                "seed": e.seed,
                "score": e.score,
                "steps": e.steps,
                "rewards": e.rewards,
                "actions": e.actions,
            }
            for e in eps
        ]
        for name, eps in all_results.items()
    }
    (out_dir / "baselines.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nRaw results written to: {out_dir / 'baselines.json'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--policies",
        nargs="+",
        default=["random", "always_call", "always_email", "rule"],
        help="Any combination of: random | always_{call,email,follow_up,ignore} | rule | llm:<model_name>",
    )
    p.add_argument("--episodes-per-tier", type=int, default=8)
    p.add_argument("--base-seed", type=int, default=1337)
    p.add_argument("--env-base-url", default="http://localhost:8000")
    p.add_argument("--api-base-url", default="http://127.0.0.1:8001/v1")
    p.add_argument("--out-dir", default="outputs/baselines")
    args = p.parse_args()
    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
