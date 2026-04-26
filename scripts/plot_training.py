"""Plot GRPO training metrics from Space stdout logs.

The trainer prints one JSON line per step. Save the Space logs to a file,
then run:

    python scripts/plot_training.py path/to/space.log

Outputs PNGs into outputs/plots/ and a TensorBoard event file at
outputs/tb-replay/ that you can view with `tensorboard --logdir outputs/tb-replay`.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def parse_lines(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        # Strip ANSI / log prefixes if any.
        line = line.strip()
        m = re.search(r"\{.*\"step\".*\}", line)
        if not m:
            continue
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if "step" in obj:
            rows.append(obj)
    rows.sort(key=lambda r: r.get("step", 0))
    return rows


def write_tb(rows, out_dir: Path) -> None:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("Install: pip install tensorboard torch", file=sys.stderr)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    w = SummaryWriter(log_dir=str(out_dir))
    for r in rows:
        step = int(r["step"])
        for k, v in r.items():
            if k == "step":
                continue
            if isinstance(v, (int, float)):
                w.add_scalar(k, float(v), step)
    w.close()
    print(f"TB events at {out_dir}  ->  tensorboard --logdir {out_dir}")


def write_pngs(rows, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install: pip install matplotlib", file=sys.stderr)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k, v in r.items()
                   if isinstance(v, (int, float)) and k != "step"})
    steps = [r["step"] for r in rows]
    for k in keys:
        ys = [r.get(k, float("nan")) for r in rows]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(steps, ys, marker=".")
        ax.set_title(k)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
        safe = k.replace("/", "_")
        fig.tight_layout()
        fig.savefig(out_dir / f"{safe}.png", dpi=120)
        plt.close(fig)
    print(f"Wrote {len(keys)} plots to {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("log", type=Path, help="Path to a file containing the trainer stdout.")
    p.add_argument("--tb-dir", type=Path, default=Path("outputs/tb-replay"))
    p.add_argument("--png-dir", type=Path, default=Path("outputs/plots"))
    args = p.parse_args()
    rows = parse_lines(args.log)
    if not rows:
        sys.exit("No JSON metric lines found.")
    print(f"Parsed {len(rows)} steps; keys: {sorted({k for r in rows for k in r})}")
    write_tb(rows, args.tb_dir)
    write_pngs(rows, args.png_dir)


if __name__ == "__main__":
    main()
