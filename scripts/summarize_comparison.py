"""Aggregate base vs trained comparison JSONs into a markdown table + bar chart.

Usage:
    python scripts/summarize_comparison.py \
        --base outputs/comparison/base.json \
        --trained outputs/comparison/trained.json \
        --out-png outputs/figures/base_vs_trained.png \
        --out-md outputs/comparison/summary.md
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


TIER_ORDER = ["easy", "medium", "hard"]


def load(path: Path):
    return json.loads(Path(path).read_text())


def per_tier(records):
    out = defaultdict(list)
    for r in records:
        out[r["tier"]].append(r)
    return out


def agg(records):
    if not records:
        return {"n": 0, "score_mean": 0.0, "success_rate": 0.0,
                "avg_total_reward": 0.0, "avg_steps": 0.0}
    scores = [r["score"] for r in records]
    rewards = [sum(r["rewards"]) for r in records]
    return {
        "n": len(records),
        "score_mean": statistics.fmean(scores),
        "success_rate": sum(1 for r in records if r["success"]) / len(records),
        "avg_total_reward": statistics.fmean(rewards),
        "avg_steps": statistics.fmean([r["steps"] for r in records]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--trained", required=True)
    ap.add_argument("--out-png", default="outputs/figures/base_vs_trained.png")
    ap.add_argument("--out-md", default="outputs/comparison/summary.md")
    args = ap.parse_args()

    base = load(Path(args.base))
    trained = load(Path(args.trained))

    base_t = per_tier(base)
    trained_t = per_tier(trained)

    rows = []
    for tier in TIER_ORDER:
        b = agg(base_t.get(tier, []))
        t = agg(trained_t.get(tier, []))
        rows.append((tier, b, t))

    # Overall.
    rows.append(("overall", agg(base), agg(trained)))

    # Markdown table.
    md = ["# Base vs Trained — Lead Triage Eval\n",
          "| Tier | n | Base score | Trained score | Δ score | Base reward | Trained reward | Δ reward |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for tier, b, t in rows:
        ds = t["score_mean"] - b["score_mean"]
        dr = t["avg_total_reward"] - b["avg_total_reward"]
        md.append(
            f"| {tier} | {b['n']} | {b['score_mean']:.3f} | {t['score_mean']:.3f} | "
            f"{ds:+.3f} | {b['avg_total_reward']:+.2f} | {t['avg_total_reward']:+.2f} | {dr:+.2f} |"
        )
    md_text = "\n".join(md) + "\n"
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(md_text)
    print(md_text)

    # Bar chart (per-tier mean grader score).
    tiers = TIER_ORDER
    base_scores = [agg(base_t.get(t, []))["score_mean"] for t in tiers]
    trained_scores = [agg(trained_t.get(t, []))["score_mean"] for t in tiers]

    x = range(len(tiers))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar([i - width / 2 for i in x], base_scores, width=width, label="Base", color="#888")
    ax.bar([i + width / 2 for i in x], trained_scores, width=width,
           label="Trained (GRPO, 50 steps)", color="#2a7ae2")
    ax.set_xticks(list(x))
    ax.set_xticklabels(tiers)
    ax.set_ylabel("Mean grader score")
    ax.set_ylim(0, 1)
    ax.set_title("Lead-Triage: base vs GRPO-trained policy")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    for i, (b, t) in enumerate(zip(base_scores, trained_scores)):
        ax.text(i - width / 2, b + 0.02, f"{b:.2f}", ha="center", fontsize=9)
        ax.text(i + width / 2, t + 0.02, f"{t:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=130)
    print(f"Wrote {args.out_png}")


if __name__ == "__main__":
    main()
