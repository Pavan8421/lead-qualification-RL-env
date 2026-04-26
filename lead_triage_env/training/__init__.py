"""Training package for the Lead Triage RL environment.

Provides prompt builders, async rollout collectors, GRPO reward shaping,
and a thin TRL/Unsloth trainer scaffold (M5–M6 of the roadmap).

Heavy ML dependencies (transformers/trl/unsloth/wandb) are imported lazily
inside the policy/trainer modules so that `prompt`, `rollout`, and `rewards`
remain importable in CI and on CPU-only machines.
"""

from .prompt import build_prompt, render_history_narrative
from .rewards import (
    EpisodeScalar,
    episode_scalar_reward,
    group_advantages,
    summarize_breakdown,
)
from .rollout import (
    EpisodeRollout,
    StepRecord,
    collect_batch,
    collect_episode,
)

__all__ = [
    "build_prompt",
    "render_history_narrative",
    "EpisodeScalar",
    "episode_scalar_reward",
    "group_advantages",
    "summarize_breakdown",
    "EpisodeRollout",
    "StepRecord",
    "collect_batch",
    "collect_episode",
]
