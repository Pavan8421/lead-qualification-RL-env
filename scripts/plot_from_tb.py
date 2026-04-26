"""Render training curves from a TensorBoard event-file directory.

Usage:
    python scripts/plot_from_tb.py outputs/adapters.grpo17/tb \
        --out outputs/figures/training_curves.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


PANELS = [
    ("train/loss", "GRPO loss"),
    ("train/kl", "KL to reference"),
    ("mean_scalar_reward", "Mean scalar reward / step"),
    ("mean_terminal_grader", "Mean terminal grader score"),
    ("conversion_rate", "Conversion rate"),
    ("mean_action_diversity", "Action diversity"),
]


def load_scalar(ea: EventAccumulator, tag: str):
    if tag not in ea.Tags()["scalars"]:
        return [], []
    events = ea.Scalars(tag)
    return [e.step for e in events], [e.value for e in events]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tb_dir")
    ap.add_argument("--out", default="outputs/figures/training_curves.png")
    ap.add_argument("--title", default="Lead-Triage GRPO — 50-step training run")
    args = ap.parse_args()

    ea = EventAccumulator(args.tb_dir)
    ea.Reload()

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (tag, title) in zip(axes.flat, PANELS):
        steps, vals = load_scalar(ea, tag)
        if not steps:
            ax.set_title(f"{title}\n(missing)")
            ax.axis("off")
            continue
        ax.plot(steps, vals, marker=".", linewidth=1.2)
        ax.set_title(title)
        ax.set_xlabel("optimizer step")
        ax.grid(alpha=0.3)

    fig.suptitle(args.title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
