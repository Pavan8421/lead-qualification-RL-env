"""Episode-level reward shaping + GRPO group advantages (roadmap §M5).

Per-step rewards (the M2 breakdown) come straight from the env. This module
collapses one episode's worth of step rewards into the scalar advantage that
GRPO needs, and standardises within a group of G completions for the same
prompt.

Anti-hack note (§8): the trainer should also receive the breakdown columns
unchanged so W&B can monitor each verifier independently. We expose a
`summarize_breakdown` helper for that.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence


@dataclass(frozen=True)
class EpisodeScalar:
    """Single-episode scalar used as the GRPO reward."""

    total_step_reward: float
    terminal_grader: float
    lambda_terminal: float

    @property
    def value(self) -> float:
        return self.total_step_reward + self.lambda_terminal * self.terminal_grader


def episode_scalar_reward(
    per_step_rewards: Iterable[float],
    terminal_grader: float,
    *,
    lambda_terminal: float = 2.0,
) -> EpisodeScalar:
    """Sum per-step rewards and add λ × terminal grader."""
    total = float(sum(per_step_rewards))
    grader = float(terminal_grader)
    return EpisodeScalar(
        total_step_reward=total,
        terminal_grader=grader,
        lambda_terminal=float(lambda_terminal),
    )


def group_advantages(
    rewards: Sequence[float],
    *,
    eps: float = 1e-6,
) -> List[float]:
    """GRPO advantage standardisation within a group of G completions.

    advantage_i = (r_i - mean(r)) / (std(r) + eps)

    For G == 1 the advantage is forced to 0 (no signal in a singleton group).
    """
    n = len(rewards)
    if n == 0:
        return []
    if n == 1:
        return [0.0]
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / n
    std = math.sqrt(var)
    return [(float(r) - mean) / (std + eps) for r in rewards]


def summarize_breakdown(
    breakdown_columns: Iterable[Mapping[str, float]],
) -> Dict[str, float]:
    """Average each verifier column across the steps of one episode.

    The trainer logs these per-episode averages to W&B so we can spot a
    column going off the rails (e.g. format_compliance collapsing to 0)
    independently of the scalar reward.
    """
    rows = list(breakdown_columns)
    if not rows:
        return {}
    keys = sorted({k for row in rows for k in row.keys()})
    summary: Dict[str, float] = {}
    for key in keys:
        values = [float(row.get(key, 0.0)) for row in rows]
        summary[key] = sum(values) / len(values)
    return summary
