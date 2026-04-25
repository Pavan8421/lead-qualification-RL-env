"""Async rollout collector for GRPO training (roadmap §M5).

`collect_episode` runs one episode against a `LeadTriageEnv` HTTP client
using a user-supplied async `policy_fn(messages, legal_tokens) -> str`. Any
completion that does not parse to a token in `legal_tokens` is *resampled*
locally — we never let the trainer absorb the −0.4 invalid-action penalty
as a reward-hacking signal (§8).

`collect_batch` fans out across many episodes/seeds with `asyncio.gather`.
GRPO's group-of-G structure is a separate concern and lives in
`rewards.group_advantages`; the rollout layer just produces episodes.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

from ..client import LeadTriageEnv
from ..models import LeadTriageAction, LeadTriageObservation
from .prompt import DEFAULT_ARG_BY_CHANNEL, build_prompt

# A policy is an async callable: (messages, legal_tokens) -> raw_completion_text.
PolicyFn = Callable[[List[Dict[str, str]], List[str]], Awaitable[str]]


@dataclass
class StepRecord:
    """One environment step inside a rollout."""

    step_index: int
    prompt_messages: List[Dict[str, str]]
    legal_tokens: List[str]
    raw_completion: str
    action_token: str
    action_payload: Dict[str, Any]
    reward: float
    done: bool
    last_event: str
    invalid_resamples: int = 0


@dataclass
class EpisodeRollout:
    """Aggregated rollout for one episode (one prompt-trajectory pair)."""

    tier: str
    seed: int
    steps: List[StepRecord] = field(default_factory=list)
    per_step_rewards: List[float] = field(default_factory=list)
    terminal_grader: float = 0.0
    converted: bool = False
    invalid_resamples_total: int = 0
    repeat_streak_max: int = 0
    last_event: str = "none"
    trajectory: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def total_step_reward(self) -> float:
        return float(sum(self.per_step_rewards))

    @property
    def num_steps(self) -> int:
        return len(self.steps)


_TOKEN_RE = re.compile(r"([A-Z_]+)(?:\(([^)]+)\))?")


def parse_action_token(text: str, legal_tokens: Sequence[str]) -> Optional[str]:
    """Try to find a legal token in the model's free-form completion.

    Strategy:
      1) Exact match against any legal token.
      2) Match `CHANNEL(arg)` regex and check membership.
      3) Match bare `CHANNEL` and synthesise default-arg token.
    Returns the canonical legal token, or `None` if nothing matched.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    legal = list(legal_tokens)
    legal_set = set(legal)

    # 1) full exact match (substring search; pick the longest that fits)
    #    Match against the original casing so lowercase argument names
    #    like `value_prop` survive; the channel names in legal tokens are
    #    already uppercase by construction.
    candidates = sorted(
        (tok for tok in legal if tok in cleaned),
        key=len,
        reverse=True,
    )
    if candidates:
        return candidates[0]

    # 2) regex pull — channels are uppercase; arguments may be mixed case.
    upper_only_for_channel_match = cleaned.upper()
    matches = list(_TOKEN_RE.finditer(upper_only_for_channel_match))
    for m in matches:
        channel = m.group(1)
        argument = m.group(2)
        if argument is not None:
            tok = f"{channel}({argument})"
            if tok in legal_set:
                return tok
        # fall back to default-arg
        default = DEFAULT_ARG_BY_CHANNEL.get(channel, None)
        if default is None:
            continue
        if channel == "IGNORE" and "IGNORE" in legal_set:
            return "IGNORE"
        synth = f"{channel}({default})" if channel != "IGNORE" else "IGNORE"
        if synth in legal_set:
            return synth
    return None


def action_payload_from_token(token: str) -> Dict[str, Any]:
    """Inverse of `prompt._flatten_legal_tokens`: build the env action dict."""
    m = _TOKEN_RE.fullmatch(token.strip())
    if not m:
        return {"channel": token.strip()}
    channel = m.group(1)
    argument = m.group(2) or DEFAULT_ARG_BY_CHANNEL.get(channel, "")
    payload: Dict[str, Any] = {"channel": channel}
    if channel == "EMAIL":
        payload["template"] = argument or "generic"
    elif channel == "CALL":
        payload["script"] = argument or "discovery"
    elif channel == "FOLLOW_UP":
        left, _, right = (argument or "email:soft").partition(":")
        payload["follow_up_channel"] = left or "email"
        payload["follow_up_tone"] = right or "soft"
    return payload


def _observation_to_dict(obs: LeadTriageObservation) -> Dict[str, Any]:
    return obs.model_dump(mode="json")


async def collect_episode(
    env: LeadTriageEnv,
    policy_fn: PolicyFn,
    *,
    seed: int,
    tier: str,
    max_resamples: int = 3,
) -> EpisodeRollout:
    """Run one episode end-to-end.

    The env contract: `reset()` -> initial observation; `step(action)` ->
    StepResult with observation/reward/done. Terminal observation carries
    `grader_score` and the trajectory log.
    """
    rollout = EpisodeRollout(tier=tier, seed=seed)
    try:
        reset_result = await env.reset(seed=seed, task_tier=tier)
    except Exception as exc:  # network / serialization failures
        rollout.error = f"reset_failed: {exc}"
        return rollout

    obs: LeadTriageObservation = reset_result.observation
    done = bool(reset_result.done)
    step_index = 0
    last_action_token: Optional[str] = None
    streak = 0

    while not done:
        step_index += 1
        obs_dict = _observation_to_dict(obs)
        prompt_bundle = build_prompt(obs_dict)
        messages = prompt_bundle["messages"]
        legal_tokens = list(prompt_bundle["legal_tokens"]) or ["IGNORE"]

        invalid_resamples = 0
        chosen_token: Optional[str] = None
        raw_completion = ""
        for _ in range(max_resamples + 1):
            raw_completion = await policy_fn(messages, legal_tokens)
            chosen_token = parse_action_token(raw_completion, legal_tokens)
            if chosen_token is not None:
                break
            invalid_resamples += 1

        if chosen_token is None:
            # Hard fallback: pick the safest legal token rather than emit a
            # malformed payload. This intentionally never reaches the env
            # with an illegal action -> no reward-hack vector.
            chosen_token = "IGNORE" if "IGNORE" in legal_tokens else legal_tokens[0]

        rollout.invalid_resamples_total += invalid_resamples
        if chosen_token == last_action_token:
            streak += 1
        else:
            streak = 1
        last_action_token = chosen_token
        rollout.repeat_streak_max = max(rollout.repeat_streak_max, streak)

        action_payload = action_payload_from_token(chosen_token)
        try:
            action_obj = LeadTriageAction(**action_payload)
        except Exception as exc:
            rollout.error = f"action_validation_failed: {exc}; payload={action_payload}"
            break

        try:
            step_result = await env.step(action_obj)
        except Exception as exc:
            rollout.error = f"step_failed: {exc}"
            break

        obs = step_result.observation
        reward = float(step_result.reward or 0.0)
        done = bool(step_result.done)
        last_event = str(getattr(obs, "last_event", "none"))

        rollout.steps.append(
            StepRecord(
                step_index=step_index,
                prompt_messages=messages,
                legal_tokens=legal_tokens,
                raw_completion=raw_completion,
                action_token=chosen_token,
                action_payload=action_payload,
                reward=reward,
                done=done,
                last_event=last_event,
                invalid_resamples=invalid_resamples,
            )
        )
        rollout.per_step_rewards.append(reward)
        rollout.last_event = last_event

        if done:
            grader = getattr(obs, "grader_score", None)
            rollout.terminal_grader = float(grader) if grader is not None else 0.0
            rollout.converted = last_event == "converted"
            traj = getattr(obs, "trajectory", None)
            if traj:
                rollout.trajectory = list(traj)
            break

    return rollout


async def collect_batch(
    env: LeadTriageEnv,
    policy_fn: PolicyFn,
    *,
    seeds_per_tier: Dict[str, Sequence[int]],
    max_concurrency: int = 4,
    max_resamples: int = 3,
) -> List[EpisodeRollout]:
    """Fan out `collect_episode` across (tier, seed) pairs."""
    sem = asyncio.Semaphore(max_concurrency)

    async def _one(tier: str, seed: int) -> EpisodeRollout:
        async with sem:
            return await collect_episode(
                env,
                policy_fn,
                seed=seed,
                tier=tier,
                max_resamples=max_resamples,
            )

    tasks: List[Awaitable[EpisodeRollout]] = []
    for tier, seeds in seeds_per_tier.items():
        for seed in seeds:
            tasks.append(_one(tier, int(seed)))
    return list(await asyncio.gather(*tasks))


# ---------- helpers exposed for scripts/inspect_rollouts.py (M7) ----------

def rollout_to_jsonable(rollout: EpisodeRollout) -> Dict[str, Any]:
    """Serializable summary suitable for `json.dump(...)`."""
    return {
        "tier": rollout.tier,
        "seed": rollout.seed,
        "num_steps": rollout.num_steps,
        "total_step_reward": rollout.total_step_reward,
        "terminal_grader": rollout.terminal_grader,
        "converted": rollout.converted,
        "invalid_resamples_total": rollout.invalid_resamples_total,
        "repeat_streak_max": rollout.repeat_streak_max,
        "last_event": rollout.last_event,
        "error": rollout.error,
        "steps": [
            {
                "step_index": s.step_index,
                "action_token": s.action_token,
                "raw_completion": s.raw_completion,
                "reward": s.reward,
                "last_event": s.last_event,
                "invalid_resamples": s.invalid_resamples,
                "legal_tokens": s.legal_tokens,
                "prompt_user": s.prompt_messages[-1]["content"]
                if s.prompt_messages
                else "",
            }
            for s in rollout.steps
        ],
        "trajectory": rollout.trajectory,
    }


# ---------- CLI: smoke-collect against a running env using a rule policy ----

async def _rule_policy(
    messages: List[Dict[str, str]], legal_tokens: List[str]
) -> str:
    """Deterministic baseline policy used by the M5 smoke CLI.

    Mirrors the conservative heuristic in `inference.py` — purposely weak so
    the LLM has clear headroom in M7.
    """
    user = messages[-1]["content"] if messages else ""
    has_prior = '"has_prior_contact":true' in user.replace(" ", "")
    if has_prior and any(t.startswith("FOLLOW_UP(") for t in legal_tokens):
        for tone in ("email:soft", "call:soft", "email:firm", "call:firm"):
            tok = f"FOLLOW_UP({tone})"
            if tok in legal_tokens:
                return tok
    for tok in ("EMAIL(generic)", "EMAIL(value_prop)", "EMAIL"):
        if tok in legal_tokens:
            return tok
    return legal_tokens[0]


async def _smoke_main(
    base_url: str,
    tier: str,
    n: int,
    base_seed: int,
    out_path: Optional[str],
) -> None:
    env = LeadTriageEnv(base_url=base_url)
    await env.connect()
    try:
        rollouts = await collect_batch(
            env,
            _rule_policy,
            seeds_per_tier={tier: [base_seed + i for i in range(n)]},
            max_concurrency=min(4, n),
        )
    finally:
        await env.close()

    payload = [rollout_to_jsonable(r) for r in rollouts]
    text = json.dumps(payload, indent=2, default=str)
    if out_path:
        from pathlib import Path

        Path(out_path).write_text(text, encoding="utf-8")
        print(f"wrote {len(rollouts)} rollouts to {out_path}")
    else:
        print(text)


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Smoke-collect rollouts using the rule baseline policy."
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="LeadTriageEnv server base URL.",
    )
    parser.add_argument("--tier", default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument("--n", type=int, default=4, help="Episodes to collect.")
    parser.add_argument("--base-seed", type=int, default=1337)
    parser.add_argument("--out", default=None, help="Optional output JSON path.")
    args = parser.parse_args()
    asyncio.run(
        _smoke_main(args.base_url, args.tier, args.n, args.base_seed, args.out)
    )


if __name__ == "__main__":
    _cli()
