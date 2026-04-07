"""Episode grader: deterministic score in [0, 1] from trajectory summary (Phase 2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .task_tier import TaskTier, TIER_CONFIGS

EPS = 1e-2


@dataclass
class EpisodeGradeInput:
    """Sufficient statistics for grading."""

    tier: TaskTier
    total_reward: float
    converted: bool
    churned: bool
    ignored: bool
    steps_taken: int
    max_steps: int
    repeat_action_streak_max: int


def _clamp_open01(x: float) -> float:
    """Clamp to strict open interval (0,1) for validator compatibility."""
    return max(EPS, min(1.0 - EPS, x))


def grade_episode(summary: EpisodeGradeInput) -> float:
    """
    Map episode statistics to [0, 1]: blend normalized return with outcome heuristic.
    """
    cfg = TIER_CONFIGS[summary.tier]
    oracle = cfg.grader_oracle_return
    rnd = cfg.grader_random_return
    span = max(1e-6, oracle - rnd)
    return_component = _clamp_open01((summary.total_reward - rnd) / span)

    heuristic = 0.35
    if summary.converted:
        heuristic += 0.55
    if summary.churned:
        heuristic -= 0.2
    if summary.repeat_action_streak_max >= 3:
        heuristic -= 0.15
    if summary.ignored and summary.steps_taken <= 1:
        heuristic -= 0.1
    if summary.steps_taken >= summary.max_steps and not summary.converted:
        heuristic -= 0.08
    heuristic = _clamp_open01(heuristic)

    score = _clamp_open01(0.62 * return_component + 0.38 * heuristic)
    # Re-clamp after rounding so we never emit 0.0 or 1.0.
    return _clamp_open01(round(score, 4))


def grade_episode_from_log(episode_log: List[Dict[str, Any]], tier: TaskTier) -> float:
    """Build summary from env trajectory records (see metadata.trajectory)."""
    total_reward = sum(float(e.get("reward", 0.0)) for e in episode_log)
    converted = any(e.get("outcome") == "converted" for e in episode_log)
    churned = any(e.get("outcome") == "churned" for e in episode_log)
    ignored = any(e.get("action") == "IGNORE" for e in episode_log)
    max_steps = int(episode_log[-1].get("max_steps", 4)) if episode_log else 4
    steps_taken = int(episode_log[-1].get("step_index", len(episode_log))) if episode_log else 0

    streak = 0
    prev: Optional[str] = None
    repeat_max = 0
    for e in episode_log:
        a = str(e.get("action", ""))
        if a == prev:
            streak += 1
        else:
            streak = 1
        repeat_max = max(repeat_max, streak)
        prev = a

    return grade_episode(
        EpisodeGradeInput(
            tier=tier,
            total_reward=total_reward,
            converted=converted,
            churned=churned,
            ignored=ignored,
            steps_taken=steps_taken,
            max_steps=max_steps,
            repeat_action_streak_max=repeat_max,
        )
    )
