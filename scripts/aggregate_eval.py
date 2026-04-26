"""Aggregate per-checkpoint eval reports into one summary table.

Reads outputs/comparison/step-NNNNNN/{base.json,trained.json} for every
checkpoint and prints a sorted table:

  step    eps   base_grader  trained_grader  delta    base_reward  trained_reward
  50      150       0.4396          0.4486   +2.1%         4.0839          4.4149
  100     150       0.4396          0.5012   +14.0%        4.0839          4.9201
  ...

Picks the best checkpoint by trained_grader and writes its name to
outputs/comparison/best_checkpoint.txt for downstream scripts.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional


def _load_eps(path: Path) -> List[dict]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except Exception:
        return []


def _agg(eps: List[dict]) -> Dict[str, float]:
    if not eps:
        return {"n": 0, "grader": 0.0, "reward": 0.0, "success": 0.0}
    scores = [float(e.get("score", 0.0)) for e in eps]
    rewards = [sum(float(r) for r in e.get("rewards", [])) for e in eps]
    return {
        "n": len(eps),
        "grader": statistics.fmean(scores),
        "reward": statistics.fmean(rewards),
        "success": sum(1 for e in eps if e.get("success")) / len(eps),
    }


def _per_tier_grader(eps: List[dict]) -> Dict[str, float]:
    by_tier: Dict[str, List[float]] = {}
    for e in eps:
        by_tier.setdefault(e.get("tier", "unknown"), []).append(float(e.get("score", 0.0)))
    return {t: statistics.fmean(v) for t, v in by_tier.items() if v}


def _action_share(eps: List[dict]) -> Dict[str, float]:
    counts: Dict[str, int] = {}
    total = 0
    for e in eps:
        for a in e.get("actions", []):
            ch = a.split("(")[0]
            counts[ch] = counts.get(ch, 0) + 1
            total += 1
    if total == 0:
        return {}
    return {k: v / total for k, v in sorted(counts.items())}


def _step_num(name: str) -> int:
    # "step-000050" -> 50
    try:
        return int(name.split("-")[-1])
    except Exception:
        return -1


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--comparison-root", default="outputs/comparison")
    p.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    args = p.parse_args()

    root = Path(args.comparison_root)
    if not root.exists():
        print(f"[FATAL] {root} does not exist", file=sys.stderr)
        sys.exit(1)

    rows = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not d.name.startswith("step-"):
            continue
        base_eps = _load_eps(d / "base.json")
        trained_eps = _load_eps(d / "trained.json")
        if not base_eps or not trained_eps:
            print(f"[skip] {d.name}: missing base.json or trained.json")
            continue
        rows.append({
            "step": _step_num(d.name),
            "name": d.name,
            "base": _agg(base_eps),
            "trained": _agg(trained_eps),
            "base_per_tier": _per_tier_grader(base_eps),
            "trained_per_tier": _per_tier_grader(trained_eps),
            "trained_actions": _action_share(trained_eps),
            "base_actions": _action_share(base_eps),
            "out_dir": d,
        })

    if not rows:
        print("[FATAL] no usable checkpoints found")
        sys.exit(1)

    rows.sort(key=lambda r: r["step"])

    # ---- main summary table -----
    print("=" * 96)
    print(f"MULTI-CHECKPOINT SUMMARY    base_model={args.base_model}")
    print("=" * 96)
    header = f"{'step':>6} {'eps':>5} {'base_grad':>10} {'tr_grad':>10} {'Δ%':>8}  {'base_rew':>9} {'tr_rew':>9} {'Δ':>8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        b_g, t_g = r["base"]["grader"], r["trained"]["grader"]
        b_r, t_r = r["base"]["reward"], r["trained"]["reward"]
        pct = (t_g - b_g) / max(1e-6, b_g) * 100 if b_g else 0.0
        d_r = t_r - b_r
        print(f"{r['step']:>6} {r['trained']['n']:>5} {b_g:>10.4f} {t_g:>10.4f} {pct:>+7.1f}%  {b_r:>9.3f} {t_r:>9.3f} {d_r:>+8.3f}")

    # ---- per-tier breakdown of best ckpt -----
    best = max(rows, key=lambda r: r["trained"]["grader"])
    print()
    print(f"BEST CHECKPOINT: {best['name']}  (trained_grader={best['trained']['grader']:.4f})")
    print()
    print("Per-tier grader at best checkpoint:")
    print(f"{'tier':<10} {'base':>10} {'trained':>10} {'Δ':>10}")
    print("-" * 42)
    tiers = sorted(set(best["base_per_tier"]) | set(best["trained_per_tier"]))
    for t in tiers:
        b = best["base_per_tier"].get(t, 0.0)
        tr = best["trained_per_tier"].get(t, 0.0)
        print(f"{t:<10} {b:>10.4f} {tr:>10.4f} {tr - b:>+10.4f}")

    # ---- action distribution shift at best ckpt -----
    print()
    print("Action distribution at best checkpoint:")
    print(f"{'action':<14} {'base':>8} {'trained':>10} {'Δpp':>10}")
    print("-" * 44)
    actions = sorted(set(best["base_actions"]) | set(best["trained_actions"]))
    for a in actions:
        b = best["base_actions"].get(a, 0.0) * 100
        tr = best["trained_actions"].get(a, 0.0) * 100
        print(f"{a:<14} {b:>7.1f}% {tr:>9.1f}% {tr - b:>+9.1f}")

    # ---- write best ckpt name for downstream scripts -----
    best_path = Path(args.comparison_root) / "best_checkpoint.txt"
    best_path.write_text(best["name"])
    print()
    print(f"-> Best checkpoint written to: {best_path}")


if __name__ == "__main__":
    main()
