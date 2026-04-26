"""M7 — Inspect rollouts + anti-hack audit.

Sampling 20 episodes from a running env, dumping full prompt/response/reward
trails for human inspection, then running the §M7 hard-fail thresholds:

  * invalid_follow_up rate > 2%        -> FAIL
  * any single action chosen > 85%     -> FAIL  (collapse / mode-seeking)
  * mean grader on `easy` <= rule base -> FAIL  (no learning)
  * action-entropy < 1.5 nats          -> FAIL  (diversity collapse)

The script can run against the rule baseline (default), the inference.py LLM,
or a trained adapter via train.py's stub policy. It does NOT compute KL —
that lives inside trainer_grpo.py during training; this script audits
*outputs*, not training-time KL.

Usage:
    python scripts/inspect_rollouts.py --tier easy --n 20 --policy stub
    python scripts/inspect_rollouts.py --policy llm  # uses OpenAI via inference.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lead_triage_env import LeadTriageEnv  # noqa: E402
from lead_triage_env.training.policy import make_stub_policy_fn  # noqa: E402
from lead_triage_env.training.rollout import (  # noqa: E402
    collect_batch,
    rollout_to_jsonable,
)


# --------------------------------------------------------------- policies ---


def _load_audit_thresholds(config_path: str) -> Dict[str, float]:
    if not Path(config_path).exists():
        return {}
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    return raw.get("audit", {}) or {}


def _make_llm_policy() -> Callable[..., Awaitable[str]]:
    """Use the same OpenAI-compatible client as inference.py."""
    import os

    from openai import OpenAI  # type: ignore

    api_key = (
        os.getenv("OPENAI_API_KEY", "").strip()
        or os.getenv("HF_TOKEN", "").strip()
        or os.getenv("API_KEY", "").strip()
    )
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY / HF_TOKEN for --policy llm.")
    base_url = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("MODEL_NAME", "gpt-4.1-mini")
    client = OpenAI(base_url=base_url, api_key=api_key)

    async def _policy(messages: List[Dict[str, str]], legal_tokens: List[str]) -> str:
        del legal_tokens

        def _call() -> str:
            completion = client.chat.completions.create(
                model=model,
                temperature=0.0,
                max_tokens=16,
                messages=messages,
            )
            return completion.choices[0].message.content or ""

        return await asyncio.to_thread(_call)

    return _policy


# ------------------------------------------------------------------ audit ---


def _entropy_nats(counts: Counter) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c == 0:
            continue
        p = c / total
        h -= p * math.log(p)
    return h


def audit_rollouts(
    rollouts: List[Dict[str, Any]],
    *,
    thresholds: Dict[str, float],
) -> Dict[str, Any]:
    """Return audit summary + list of failures (empty list == pass)."""
    action_counts: Counter = Counter()
    invalid_count = 0
    total_steps = 0
    grader_scores: List[float] = []
    repeat_streaks: List[int] = []
    conversion = 0

    for r in rollouts:
        for s in r.get("steps", []):
            action_counts[s.get("action_token", "?")] += 1
            invalid_count += int(s.get("invalid_resamples", 0))
            total_steps += 1
        grader_scores.append(float(r.get("terminal_grader", 0.0)))
        repeat_streaks.append(int(r.get("repeat_streak_max", 0)))
        if r.get("converted"):
            conversion += 1

    invalid_rate = invalid_count / max(1, total_steps)
    most_common_share = (
        action_counts.most_common(1)[0][1] / total_steps if total_steps else 0.0
    )
    entropy = _entropy_nats(action_counts)
    mean_grader = sum(grader_scores) / max(1, len(grader_scores))

    failures: List[str] = []
    inv_max = float(thresholds.get("invalid_follow_up_rate_max", 0.02))
    share_max = float(thresholds.get("single_action_share_max", 0.85))
    ent_min = float(thresholds.get("action_entropy_min_nats", 1.5))
    rule_base = float(thresholds.get("rule_baseline_grader_easy", 0.45))

    if invalid_rate > inv_max:
        failures.append(
            f"invalid_resample_rate={invalid_rate:.3f} > {inv_max:.3f}"
        )
    if most_common_share > share_max:
        top = action_counts.most_common(1)[0][0]
        failures.append(
            f"single_action_share={most_common_share:.3f} > {share_max:.3f} "
            f"(top={top!r})"
        )
    if entropy < ent_min:
        failures.append(
            f"action_entropy_nats={entropy:.3f} < {ent_min:.3f}"
        )
    if mean_grader <= rule_base:
        failures.append(
            f"mean_grader={mean_grader:.3f} <= rule_baseline={rule_base:.3f}"
        )

    return {
        "n_episodes": len(rollouts),
        "n_steps": total_steps,
        "mean_grader": mean_grader,
        "conversion_rate": conversion / max(1, len(rollouts)),
        "invalid_resample_rate": invalid_rate,
        "action_distribution": dict(action_counts),
        "single_action_share_top": most_common_share,
        "action_entropy_nats": entropy,
        "mean_repeat_streak_max": (
            sum(repeat_streaks) / max(1, len(repeat_streaks))
        ),
        "thresholds": thresholds,
        "failures": failures,
        "passed": not failures,
    }


# -------------------------------------------------------------------- cli ---


async def _async_main(cli: argparse.Namespace) -> int:
    if cli.policy == "stub":
        policy_fn = make_stub_policy_fn()
    elif cli.policy == "llm":
        policy_fn = _make_llm_policy()
    else:
        raise SystemExit(f"unknown --policy {cli.policy}")

    seeds = [cli.base_seed + i for i in range(cli.n)]
    env = LeadTriageEnv(base_url=cli.env_base_url)
    await env.connect()
    try:
        rollouts = await collect_batch(
            env,
            policy_fn,
            seeds_per_tier={cli.tier: seeds},
            max_concurrency=min(cli.concurrency, cli.n),
        )
    finally:
        await env.close()

    payload = [rollout_to_jsonable(r) for r in rollouts]
    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rollout_path = out_dir / "rollouts.json"
    rollout_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    thresholds = _load_audit_thresholds(cli.config)
    audit = audit_rollouts(payload, thresholds=thresholds)
    audit_path = out_dir / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")

    print(json.dumps(audit, indent=2, default=str))
    print(f"\n[INSPECT] wrote {len(payload)} rollouts to {rollout_path}")
    print(f"[INSPECT] wrote audit to {audit_path}")

    if not audit["passed"]:
        print("\n[AUDIT] FAILURES:", file=sys.stderr)
        for f in audit["failures"]:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("\n[AUDIT] PASSED")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample rollouts and run M7 anti-hack audit."
    )
    parser.add_argument(
        "--policy", choices=["stub", "llm"], default="stub",
    )
    parser.add_argument("--tier", default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument("--n", type=int, default=20, help="Episodes to sample.")
    parser.add_argument("--base-seed", type=int, default=9000)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--env-base-url", default="http://localhost:8000")
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--out-dir", default="outputs/inspect")
    args = parser.parse_args()
    sys.exit(asyncio.run(_async_main(args)))


if __name__ == "__main__":
    main()
