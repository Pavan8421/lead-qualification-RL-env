"""GRPO trainer scaffold (roadmap §M6).

This module wires the env-grounded rollout (`rollout.collect_episode`) into
TRL's `GRPOTrainer`. Each TRL "reward function" gets the same rollout result
and projects out one verifier column; that way W&B logs each M2 column
independently (§15) instead of one composite scalar.

Heavy deps (`trl`, `wandb`, `transformers`) are imported lazily so importing
this module on a CPU-only box is fine for static analysis.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from ..client import LeadTriageEnv
from .policy import GRPOLLMPolicy, PolicyConfig
from .prompt import build_prompt
from .rewards import episode_scalar_reward
from .rollout import EpisodeRollout, collect_episode


@dataclass
class GRPOConfig:
    """Trainer-level config loaded from `configs/training.yaml::training`."""

    output_dir: str = "outputs/adapters"
    learning_rate: float = 5e-6
    num_generations: int = 4  # G
    kl_beta: float = 0.04
    episodes_per_step: int = 8
    total_optimizer_steps: int = 50
    lambda_terminal_grader: float = 2.0
    save_every_steps: int = 25
    eval_every_steps: int = 25
    seed: int = 1337
    tier_mix: Sequence[str] = field(default_factory=lambda: ("easy",))
    env_base_url: str = "http://localhost:8000"
    env_concurrency: int = 4
    log_to_wandb: bool = True
    wandb_project: str = "lead-triage-rl"


# A reward function in TRL has the signature:
#   reward_func(prompts, completions, **kwargs) -> List[float]
# where prompts/completions are aligned lists of length B*G.
# We attach the rollout result for each (prompt, completion) pair via kwargs.
RewardFunc = Callable[..., List[float]]


def _make_column_reward_fn(column: str) -> RewardFunc:
    """Build a TRL reward function that projects one verifier column.

    Expects each completion to carry a `rollout` attribute (added by the
    sampling hook) — a `dict` with the breakdown summary.
    """

    def _fn(prompts: List[Any], completions: List[Any], **kwargs: Any) -> List[float]:
        rollouts: List[Optional[Mapping[str, Any]]] = kwargs.get("rollouts") or []
        out: List[float] = []
        for i, _completion in enumerate(completions):
            r = rollouts[i] if i < len(rollouts) else None
            if not r:
                out.append(0.0)
                continue
            if column == "scalar":
                out.append(float(r.get("scalar_reward", 0.0)))
            else:
                out.append(float(r.get(column, 0.0)))
        return out

    _fn.__name__ = f"reward_{column}"
    return _fn


def build_reward_funcs() -> List[RewardFunc]:
    """Return the per-column reward functions registered with the trainer.

    The first one is the scalar that drives the gradient; the others are
    diagnostic and contribute zero weight in the trainer config.
    """
    return [
        _make_column_reward_fn("scalar"),
        _make_column_reward_fn("terminal_grader"),
        _make_column_reward_fn("total_step_reward"),
        _make_column_reward_fn("converted"),
        _make_column_reward_fn("invalid_resamples_total"),
        _make_column_reward_fn("repeat_streak_max"),
    ]


def rollout_to_reward_dict(
    rollout: EpisodeRollout, *, lambda_terminal: float
) -> Dict[str, float]:
    """Flatten `EpisodeRollout` -> the dict consumed by `_make_column_reward_fn`."""
    scalar = episode_scalar_reward(
        rollout.per_step_rewards,
        rollout.terminal_grader,
        lambda_terminal=lambda_terminal,
    )
    return {
        "scalar_reward": scalar.value,
        "terminal_grader": rollout.terminal_grader,
        "total_step_reward": rollout.total_step_reward,
        "converted": 1.0 if rollout.converted else 0.0,
        "invalid_resamples_total": float(rollout.invalid_resamples_total),
        "repeat_streak_max": float(rollout.repeat_streak_max),
    }


# ---------------------------------------------------------------------------
# Trainer entrypoint
# ---------------------------------------------------------------------------


class LeadTriageGRPOTrainer:
    """End-to-end GRPO trainer wired against a running LeadTriageEnv.

    Notes
    -----
    * One GRPO "prompt" == one episode reset. We collect G rollouts per prompt
      (group-of-G), turn each into one scalar via `episode_scalar_reward`, then
      hand the standardised group advantages to TRL.
    * The loop here is intentionally thin: most of the heavy lifting lives in
      the rollout collector (reusable, testable) and TRL's `GRPOTrainer`.
    """

    def __init__(
        self,
        config: GRPOConfig,
        policy: Optional[GRPOLLMPolicy] = None,
        *,
        policy_config: Optional[PolicyConfig] = None,
    ) -> None:
        self.config = config
        self.policy = policy or GRPOLLMPolicy(policy_config)
        self._env: Optional[LeadTriageEnv] = None

    # ----- env lifecycle ------------------------------------------------

    async def _ensure_env(self) -> LeadTriageEnv:
        if self._env is None:
            env = LeadTriageEnv(base_url=self.config.env_base_url)
            await env.connect()
            self._env = env
        return self._env

    async def close(self) -> None:
        if self._env is not None:
            await self._env.close()
            self._env = None

    # ----- one optimizer step -------------------------------------------

    async def _collect_group(
        self, *, seed: int, tier: str, group_size: int
    ) -> List[EpisodeRollout]:
        env = await self._ensure_env()
        # If the policy supports id capture, use it so grpo_update can
        # re-score the exact same tokens that produced the rollout.
        if hasattr(self.policy, "as_async_policy_fn"):
            try:
                policy_fn = self.policy.as_async_policy_fn(capture_ids=True)
            except TypeError:
                policy_fn = self.policy.as_async_policy_fn()
        else:
            raise RuntimeError("policy must expose `as_async_policy_fn`")
        sem = asyncio.Semaphore(self.config.env_concurrency)

        async def _one(g_idx: int) -> EpisodeRollout:
            async with sem:
                # Vary seed within the group so the G rollouts actually differ
                # in stochastic-outcome realisations (otherwise advantages
                # collapse to 0).
                return await collect_episode(
                    env, policy_fn, seed=seed * 100 + g_idx, tier=tier
                )

        return list(await asyncio.gather(*[_one(i) for i in range(group_size)]))

    # ----- public API ---------------------------------------------------

    async def step_once(self, *, step_index: int) -> Dict[str, float]:
        """Run a single optimizer step (collect group + apply GRPO gradient).

        Pipeline:
          1. Pick a tier from `tier_mix`.
          2. Collect G env episodes with the current policy (sampling
             captures token ids via `policy.sample_with_ids`).
          3. Compute the GRPO group advantages over the G episode scalars.
          4. For each step in each rollout, broadcast the rollout's
             advantage to that step's `sample_handle` (so every action
             token contributes a policy-gradient term).
          5. Call `policy.grpo_update(handles, advantages)` -> one
             AdamW step on the LoRA params.
        """
        from .rewards import group_advantages

        tier = self.config.tier_mix[step_index % len(self.config.tier_mix)]
        rollouts = await self._collect_group(
            seed=self.config.seed + step_index,
            tier=tier,
            group_size=self.config.num_generations,
        )

        rewards = [
            rollout_to_reward_dict(r, lambda_terminal=self.config.lambda_terminal_grader)
            for r in rollouts
        ]
        scalars = [r["scalar_reward"] for r in rewards]
        advantages = group_advantages(scalars)

        # Broadcast each episode's advantage to its constituent steps.
        handles: List[int] = []
        per_step_adv: List[float] = []
        for rollout, adv in zip(rollouts, advantages):
            for step in rollout.steps:
                if step.sample_handle is None:
                    continue
                handles.append(int(step.sample_handle))
                per_step_adv.append(float(adv))

        update_metrics: Dict[str, float] = {}
        if handles and hasattr(self.policy, "grpo_update"):
            update_metrics = self.policy.grpo_update(handles, per_step_adv)
        elif hasattr(self.policy, "release_handles"):
            # Eval / stub path — drop any cached ids without a gradient step.
            self.policy.release_handles(handles)

        metrics: Dict[str, float] = {
            "step": float(step_index),
            "tier": float({"easy": 0, "medium": 1, "hard": 2}.get(tier, 0)),
            "mean_scalar_reward": float(sum(scalars) / max(1, len(scalars))),
            "mean_terminal_grader": float(
                sum(r["terminal_grader"] for r in rewards) / max(1, len(rewards))
            ),
            "mean_total_step_reward": float(
                sum(r["total_step_reward"] for r in rewards) / max(1, len(rewards))
            ),
            "conversion_rate": float(
                sum(r["converted"] for r in rewards) / max(1, len(rewards))
            ),
            "mean_invalid_resamples": float(
                sum(r["invalid_resamples_total"] for r in rewards) / max(1, len(rewards))
            ),
            "max_repeat_streak": float(max((r["repeat_streak_max"] for r in rewards), default=0)),
            "advantage_std": float(_std(advantages)),
            "n_grad_samples": float(len(handles)),
        }
        for k, v in update_metrics.items():
            metrics[f"train/{k}"] = v
        return metrics


def _std(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    mean = sum(xs) / len(xs)
    return (sum((x - mean) ** 2 for x in xs) / len(xs)) ** 0.5
