"""Compare base vs trained model on the lead-triage env.

Runs `inference.py` twice (once per MODEL_NAME), parses the [START]/[STEP]/[END]
log lines, and writes a side-by-side metrics report.

Prereqs:
  * Env server running on http://localhost:8000
  * vLLM (or any OpenAI-compatible server) running with BOTH models loaded
    under the same API. With vLLM:

      python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --enable-lora \
        --lora-modules trained=./outputs/adapters/step-000050 \
        --port 8001

    -> base model name = "Qwen/Qwen2.5-1.5B-Instruct"
    -> trained model name = "trained"

Usage:
  python scripts/compare_models.py \
    --base-model "Qwen/Qwen2.5-1.5B-Instruct" \
    --trained-model "trained" \
    --episodes-per-tier 4 \
    --api-base-url http://127.0.0.1:8001/v1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


RE_START = re.compile(r"^\[START\] task=(\S+) env=\S+ model=(\S+)")
RE_STEP = re.compile(
    r"^\[STEP\] step=(\d+) action=(\S+) reward=(-?\d+\.\d+) done=(true|false) error=(.+)$"
)
RE_END = re.compile(
    r"^\[END\] success=(true|false) steps=(\d+) score=(-?\d+\.\d+) rewards=(.*)$"
)
RE_TIER = re.compile(r"lead_triage-(easy|medium|hard)-\d+")


@dataclass
class Episode:
    task: str
    model: str
    tier: str
    success: bool = False
    steps: int = 0
    score: float = 0.0
    rewards: List[float] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    errors: int = 0


def _parse_log(text: str) -> List[Episode]:
    episodes: List[Episode] = []
    cur: Optional[Episode] = None
    for line in text.splitlines():
        m = RE_START.match(line)
        if m:
            task, model = m.group(1), m.group(2)
            tier_m = RE_TIER.search(task)
            cur = Episode(
                task=task,
                model=model,
                tier=tier_m.group(1) if tier_m else "unknown",
            )
            continue
        if cur is None:
            continue
        m = RE_STEP.match(line)
        if m:
            cur.actions.append(m.group(2))
            err = m.group(5).strip()
            if err and err != "null":
                cur.errors += 1
            continue
        m = RE_END.match(line)
        if m:
            cur.success = m.group(1) == "true"
            cur.steps = int(m.group(2))
            cur.score = float(m.group(3))
            raw_rewards = m.group(4).strip()
            if raw_rewards:
                cur.rewards = [float(x) for x in raw_rewards.split(",")]
            episodes.append(cur)
            cur = None
    return episodes


def _run_inference(
    model_name: str,
    api_base_url: str,
    episodes_per_tier: int,
    base_seed: int,
    extra_env: Dict[str, str],
    repo_root: Path,
) -> Tuple[str, int]:
    env = os.environ.copy()
    env.update(
        {
            "MODEL_NAME": model_name,
            "API_BASE_URL": api_base_url,
            "EPISODES_PER_TIER": str(episodes_per_tier),
            "BASE_SEED": str(base_seed),
            "HF_TOKEN": env.get("HF_TOKEN", "dummy"),
        }
    )
    env.update(extra_env)
    print(f"\n=== Running inference: model={model_name} ===", flush=True)
    proc = subprocess.run(
        [sys.executable, "inference.py"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )
    # Surface stderr (debug noise) but don't fail on non-zero — partial logs are useful.
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    return proc.stdout, proc.returncode


# ---------- metrics ----------------------------------------------------------


def _agg(eps: List[Episode]) -> Dict[str, float]:
    if not eps:
        return {
            "n": 0,
            "score_mean": 0.0,
            "score_median": 0.0,
            "score_std": 0.0,
            "success_rate": 0.0,
            "avg_steps": 0.0,
            "avg_total_reward": 0.0,
            "error_rate": 0.0,
        }
    scores = [e.score for e in eps]
    rewards = [sum(e.rewards) for e in eps]
    return {
        "n": len(eps),
        "score_mean": statistics.fmean(scores),
        "score_median": statistics.median(scores),
        "score_std": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
        "success_rate": sum(1 for e in eps if e.success) / len(eps),
        "avg_steps": statistics.fmean([e.steps for e in eps]),
        "avg_total_reward": statistics.fmean(rewards),
        "error_rate": sum(e.errors for e in eps) / max(1, sum(e.steps for e in eps)),
    }


def _action_distribution(eps: List[Episode]) -> Dict[str, float]:
    counts: Dict[str, int] = {}
    total = 0
    for e in eps:
        for a in e.actions:
            channel = a.split("(")[0]
            counts[channel] = counts.get(channel, 0) + 1
            total += 1
    if total == 0:
        return {}
    return {k: v / total for k, v in sorted(counts.items())}


def _by_tier(eps: List[Episode]) -> Dict[str, List[Episode]]:
    out: Dict[str, List[Episode]] = {}
    for e in eps:
        out.setdefault(e.tier, []).append(e)
    return out


# ---------- formatting -------------------------------------------------------


def _fmt_pct(x: float) -> str:
    return f"{x * 100:5.1f}%"


def _delta(new: float, old: float) -> str:
    d = new - old
    arrow = "↑" if d > 1e-6 else ("↓" if d < -1e-6 else "·")
    return f"{arrow} {d:+.3f}"


def _print_report(base: List[Episode], trained: List[Episode]) -> None:
    print("\n" + "=" * 78)
    print("LEAD-TRIAGE: BASE vs TRAINED — comparison report")
    print("=" * 78)

    bm = _agg(base)
    tm = _agg(trained)

    rows = [
        ("episodes (n)", bm["n"], tm["n"]),
        ("grader score (mean)", bm["score_mean"], tm["score_mean"]),
        ("grader score (median)", bm["score_median"], tm["score_median"]),
        ("grader score (std)", bm["score_std"], tm["score_std"]),
        ("success rate", bm["success_rate"], tm["success_rate"]),
        ("avg total reward", bm["avg_total_reward"], tm["avg_total_reward"]),
        ("avg steps / ep", bm["avg_steps"], tm["avg_steps"]),
        ("error rate (per step)", bm["error_rate"], tm["error_rate"]),
    ]

    print(f"\n{'metric':<24}{'BASE':>14}{'TRAINED':>14}    delta")
    print("-" * 78)
    for name, b, t in rows:
        if name == "episodes (n)":
            print(f"{name:<24}{int(b):>14d}{int(t):>14d}")
        else:
            print(f"{name:<24}{b:>14.4f}{t:>14.4f}    {_delta(t, b)}")

    # Per-tier breakdown
    print("\n--- per-tier mean grader score ---")
    bt = _by_tier(base)
    tt = _by_tier(trained)
    tiers = sorted(set(bt) | set(tt))
    print(f"{'tier':<10}{'BASE':>12}{'TRAINED':>12}    delta")
    for tier in tiers:
        b = _agg(bt.get(tier, []))["score_mean"]
        t = _agg(tt.get(tier, []))["score_mean"]
        print(f"{tier:<10}{b:>12.4f}{t:>12.4f}    {_delta(t, b)}")

    # Action distribution
    print("\n--- action distribution (channel only) ---")
    ba = _action_distribution(base)
    ta = _action_distribution(trained)
    keys = sorted(set(ba) | set(ta))
    print(f"{'action':<14}{'BASE':>10}{'TRAINED':>10}    delta")
    for k in keys:
        b = ba.get(k, 0.0)
        t = ta.get(k, 0.0)
        print(f"{k:<14}{_fmt_pct(b):>10}{_fmt_pct(t):>10}    {_delta(t, b)}")

    # Headline verdict
    print("\n--- verdict ---")
    delta_score = tm["score_mean"] - bm["score_mean"]
    delta_reward = tm["avg_total_reward"] - bm["avg_total_reward"]
    pct = (delta_score / max(1e-6, bm["score_mean"])) * 100 if bm["score_mean"] else 0.0
    direction = "improved" if delta_score > 0 else ("regressed" if delta_score < 0 else "unchanged")
    print(
        f"Trained model {direction}: grader mean {bm['score_mean']:.4f} -> "
        f"{tm['score_mean']:.4f}  ({pct:+.1f}%); "
        f"avg reward Δ {delta_reward:+.3f}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", required=True)
    p.add_argument("--trained-model", required=True)
    p.add_argument("--api-base-url", default="http://127.0.0.1:8001/v1")
    p.add_argument("--episodes-per-tier", type=int, default=4)
    p.add_argument("--base-seed", type=int, default=1337)
    p.add_argument(
        "--out-dir",
        default="outputs/comparison",
        help="Directory for raw logs + parsed JSON.",
    )
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    extra_env = {"LEAD_TRIAGE_ENV_BASE_URL": os.environ.get("LEAD_TRIAGE_ENV_BASE_URL", "http://localhost:8000")}

    base_log, _ = _run_inference(
        args.base_model, args.api_base_url, args.episodes_per_tier,
        args.base_seed, extra_env, repo_root,
    )
    (out_dir / "base.log").write_text(base_log, encoding="utf-8")

    trained_log, _ = _run_inference(
        args.trained_model, args.api_base_url, args.episodes_per_tier,
        args.base_seed, extra_env, repo_root,
    )
    (out_dir / "trained.log").write_text(trained_log, encoding="utf-8")

    base_eps = _parse_log(base_log)
    trained_eps = _parse_log(trained_log)

    (out_dir / "base.json").write_text(
        json.dumps([e.__dict__ for e in base_eps], indent=2), encoding="utf-8"
    )
    (out_dir / "trained.json").write_text(
        json.dumps([e.__dict__ for e in trained_eps], indent=2), encoding="utf-8"
    )

    _print_report(base_eps, trained_eps)
    print(f"\nRaw logs + parsed JSON written to: {out_dir}")


if __name__ == "__main__":
    main()
