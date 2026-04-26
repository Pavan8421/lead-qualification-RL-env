"""
Inference script — Lead Triage OpenEnv (hackathon contract)
===========================================================
MANDATORY (configure before submit):
    API_BASE_URL     LLM API endpoint (OpenAI-compatible).
    MODEL_NAME       Model id for chat completions.
    HF_TOKEN         Hugging Face / API key (used as api_key for OpenAI client).
    LOCAL_IMAGE_NAME Optional. If set, env is started via from_docker_image().

Defaults are applied ONLY for API_BASE_URL and MODEL_NAME.

STDOUT must emit only:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

# Disable websocket keepalive pings before importing the env client.
# CPU-vLLM generation can exceed the default 20s ping timeout and drop the WS.
import openenv.core.env_client as _oec
_orig_ws_connect = _oec.ws_connect
def _patched_ws_connect(*args, **kwargs):
    kwargs.setdefault("ping_interval", None)
    kwargs.setdefault("ping_timeout", None)
    return _orig_ws_connect(*args, **kwargs)
_oec.ws_connect = _patched_ws_connect

from lead_triage_env import LeadTriageAction, LeadTriageEnv, grade_episode_from_log


TIERS: List[str] = ["easy", "medium", "hard"]
ALL_ACTIONS: List[str] = ["CALL", "EMAIL", "FOLLOW_UP", "IGNORE"]
DEFAULT_ARG_BY_CHANNEL: Dict[str, str] = {
    "EMAIL": "generic",
    "CALL": "discovery",
    "FOLLOW_UP": "email:soft",
    "IGNORE": "",
}
EPS = 1e-2


def _load_dotenv_if_present() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        row = line.strip()
        if not row or row.startswith("#") or "=" not in row:
            continue
        key, value = row.split("=", 1)
        k = key.strip()
        if not k:
            continue
        v = value.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp_open01(x: float) -> float:
    return max(EPS, min(1.0 - EPS, x))


def _extract_grader_score(
    grader_score: object,
    trajectory: object,
    metadata: object,
    tier: str,
) -> float:
    explicit = _safe_float(grader_score, -1.0)
    if explicit >= 0.0:
        return explicit
    if isinstance(trajectory, list) and trajectory:
        try:
            return _safe_float(grade_episode_from_log(trajectory, tier), 0.0)
        except Exception:
            return 0.0
    if isinstance(metadata, dict):
        raw = metadata.get("grader_score")
        if raw is not None:
            return _safe_float(raw, 0.0)
        traj = metadata.get("trajectory")
        if isinstance(traj, list) and traj:
            try:
                return _safe_float(grade_episode_from_log(traj, tier), 0.0)
            except Exception:
                return 0.0
    return EPS


def _observation_error(obs: object) -> Optional[str]:
    meta = getattr(obs, "metadata", None)
    if isinstance(meta, dict):
        err = meta.get("error") or meta.get("last_action_error")
        if err is not None:
            return str(err)
    return None


def _extract_action(raw: str, legal_actions: List[str]) -> str:
    txt = (raw or "").upper()
    for act in legal_actions:
        if act.upper() in txt:
            return act
    return "EMAIL" if "EMAIL" in legal_actions else legal_actions[0]


def _flatten_legal_tokens(
    legal_action_map: Dict[str, List[str]] | None,
    legal_actions: List[str],
) -> List[str]:
    if not legal_action_map:
        return list(legal_actions)
    tokens: List[str] = []
    for channel, args in legal_action_map.items():
        if channel == "IGNORE":
            tokens.append("IGNORE")
            continue
        for arg in args:
            tokens.append(f"{channel}({arg})")
    return tokens


def _parse_action_token(token: str) -> Tuple[str, str]:
    txt = token.strip()
    m = re.match(r"^([A-Z_]+)\(([^)]+)\)$", txt)
    if m:
        return m.group(1), m.group(2)
    return txt, DEFAULT_ARG_BY_CHANNEL.get(txt, "")


def _build_action_payload(channel: str, argument: str) -> Dict[str, object]:
    payload: Dict[str, object] = {"channel": channel}
    if channel == "EMAIL":
        payload["template"] = argument or "generic"
    elif channel == "CALL":
        payload["script"] = argument or "discovery"
    elif channel == "FOLLOW_UP":
        left, _, right = (argument or "email:soft").partition(":")
        payload["follow_up_channel"] = left or "email"
        payload["follow_up_tone"] = right or "soft"
    return payload


_INTENT_ESTIMATE_TO_SCORE: Dict[str, float] = {
    "low": 0.20,
    "medium": 0.50,
    "high": 0.80,
    "unknown": 0.50,
}


def _effective_intent(observation: Dict[str, object]) -> float:
    """Use intent_score if exposed; fall back to intent_estimate when env masks it (v2)."""
    raw = _safe_float(observation.get("intent_score"), -1.0)
    has_prior = bool(observation.get("has_prior_contact", False))
    if raw > 0.0 or has_prior:
        return max(0.0, raw) if raw >= 0.0 else _safe_float(observation.get("engagement_score"), 0.5)
    estimate = str(observation.get("intent_estimate", "unknown")).lower()
    if estimate in _INTENT_ESTIMATE_TO_SCORE:
        base = _INTENT_ESTIMATE_TO_SCORE[estimate]
    else:
        base = _safe_float(observation.get("engagement_score"), 0.5)

    history_summary = observation.get("history_summary")
    if isinstance(history_summary, dict):
        inbound_count = int(_safe_float(history_summary.get("inbound_count"), 0))
        last_inbound_sentiment = _safe_float(
            history_summary.get("last_inbound_sentiment"), 0.0
        )
        silence_gap = int(_safe_float(history_summary.get("longest_silence_gap"), 0))
        # Keep rule baseline simple: a tiny adjustment from history_summary scalars only.
        base += min(0.08, inbound_count * 0.015)
        base += 0.08 * max(-1.0, min(1.0, last_inbound_sentiment))
        if silence_gap >= 12:
            base -= 0.06
    return max(0.0, min(1.0, base))


def _rule_policy_action(observation: Dict[str, object], legal_actions: List[str]) -> str:
    """Deterministic fallback policy tuned for lead triage structure."""
    intent = _effective_intent(observation)
    attempts = int(_safe_float(observation.get("contact_attempts"), 0))
    step_index = int(_safe_float(observation.get("step_index"), 0))
    max_steps = int(_safe_float(observation.get("max_steps"), 4))
    has_prior = bool(observation.get("has_prior_contact", False))
    urgency = str(observation.get("urgency_level", "medium")).lower()
    deal_value = _safe_float(observation.get("estimated_deal_value"), 0.0)

    if "IGNORE" in legal_actions and attempts >= 3 and intent < 0.25:
        return "IGNORE"

    if not has_prior:
        if "CALL" in legal_actions and (intent >= 0.75 or urgency == "high" or deal_value >= 20000.0):
            return "CALL"
        if "EMAIL" in legal_actions:
            return "EMAIL"
    else:
        if "FOLLOW_UP" in legal_actions and (intent >= 0.55 or step_index >= max_steps - 1):
            return "FOLLOW_UP"
        if intent >= 0.65 and "CALL" in legal_actions:
            return "CALL"
        if "EMAIL" in legal_actions:
            return "EMAIL"

    return legal_actions[0]


def _select_final_action(
    llm_action: str,
    observation: Dict[str, object],
    legal_actions: List[str],
) -> str:
    """Use LLM action with guardrails; fallback to deterministic rule policy."""
    if llm_action not in legal_actions:
        return _rule_policy_action(observation, legal_actions)

    # Escape hatch: bypass rule guardrails so eval reflects raw LLM policy.
    if os.environ.get("INFERENCE_DISABLE_GUARDRAILS") == "1":
        return llm_action

    intent = _effective_intent(observation)
    attempts = int(_safe_float(observation.get("contact_attempts"), 0))
    has_prior = bool(observation.get("has_prior_contact", False))

    # Avoid repeatedly expensive outreach on clearly weak leads.
    if llm_action == "CALL" and intent < 0.3 and attempts >= 1:
        if "EMAIL" in legal_actions:
            return "EMAIL"
        if "IGNORE" in legal_actions:
            return "IGNORE"

    # Prefer FOLLOW_UP once contact exists and signal is decent.
    if has_prior and intent >= 0.55 and "FOLLOW_UP" in legal_actions and llm_action in ("CALL", "EMAIL"):
        return "FOLLOW_UP"

    return llm_action


def log_start(task: str, env_name: str, model: str) -> None:
    print(
        f"[START] task={task} env={env_name} model={model}",
        flush=True,
    )


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    if error:
        err = error.replace("\r", " ").replace("\n", " ").strip()
    else:
        err = "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def choose_action(
    client: OpenAI,
    model_name: str,
    observation: Dict[str, object],
    legal_actions: List[str],
    legal_action_map: Dict[str, List[str]] | None = None,
) -> tuple[Dict[str, object], str, Optional[str]]:
    legal_tokens = _flatten_legal_tokens(legal_action_map, legal_actions)
    prompt_payload = {
        "observation": observation,
        "legal_actions": legal_tokens,
        "instruction": "Return exactly one action token from legal_actions.",
    }
    try:
        completion = client.chat.completions.create(
            model=model_name,
            temperature=float(os.environ.get("INFERENCE_TEMPERATURE", "0")),
            max_tokens=8,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a lead triage policy. "
                        "Reply with exactly one token from legal_actions; if arguments are present, "
                        "reply as CHANNEL(argument). "
                        "Use only a legal action."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(prompt_payload, separators=(",", ":")),
                },
            ],
        )
        content = completion.choices[0].message.content or ""
        llm_action = _extract_action(content, legal_tokens)
        llm_channel, llm_arg = _parse_action_token(llm_action)
        candidate_channel = _select_final_action(llm_channel, observation, legal_actions)
        if candidate_channel != llm_channel:
            final_channel = candidate_channel
            final_arg = DEFAULT_ARG_BY_CHANNEL.get(final_channel, "")
        else:
            final_channel = llm_channel
            final_arg = llm_arg or DEFAULT_ARG_BY_CHANNEL.get(final_channel, "")
        final_token = (
            f"{final_channel}({final_arg})" if final_channel != "IGNORE" else "IGNORE"
        )
        if final_token not in legal_tokens:
            # Keep backward compatibility: accept bare channel when legal map absent.
            if legal_action_map:
                for token in legal_tokens:
                    if token.startswith(f"{final_channel}(") or token == final_channel:
                        final_token = token
                        break
        channel, arg = _parse_action_token(final_token)
        return _build_action_payload(channel, arg), final_token, None
    except Exception as exc:
        fallback = _rule_policy_action(observation, legal_actions)
        arg = DEFAULT_ARG_BY_CHANNEL.get(fallback, "")
        fallback_token = f"{fallback}({arg})" if fallback != "IGNORE" else "IGNORE"
        return _build_action_payload(fallback, arg), fallback_token, str(exc)


async def run_episode(
    env: LeadTriageEnv,
    client: OpenAI,
    model_name: str,
    task_name: str,
    benchmark: str,
    tier: str,
    seed: int,
    ) -> None:
    log_start(task=task_name, env_name=benchmark, model=model_name)

    reset_result = await env.reset(seed=seed, task_tier=tier)
    obs = reset_result.observation
    done = bool(reset_result.done)
    step_n = 0
    rewards: List[float] = []
    final_score = EPS
    episode_success = True

    while not done:
        step_n += 1
        obs_dict = obs.model_dump(mode="json")
        legal = list(obs.legal_actions or ALL_ACTIONS)
        legal_map = getattr(obs, "legal_action_map", None)
        action_payload, action_token, llm_err = await asyncio.to_thread(
            choose_action,
            client,
            model_name,
            obs_dict,
            legal,
            legal_map if isinstance(legal_map, dict) else None,
        )

        step_result = await env.step(LeadTriageAction(**action_payload))
        obs = step_result.observation
        done = bool(step_result.done)
        reward = _safe_float(step_result.reward)
        rewards.append(reward)

        env_err = _observation_error(obs)
        err_out = env_err if env_err is not None else llm_err
        if err_out is not None:
            episode_success = False

        log_step(
            step=step_n,
            action=action_token,
            reward=reward,
            done=done,
            error=err_out,
        )

        if done:
            final_score = _extract_grader_score(
                getattr(obs, "grader_score", None),
                getattr(obs, "trajectory", None),
                getattr(obs, "metadata", None),
                tier,
            )
            final_score = _clamp_open01(final_score)

    log_end(
        success=episode_success and (final_score > 0.0 and final_score < 1.0),
        steps=step_n,
        score=final_score,
        rewards=rewards,
    )


async def _async_main() -> None:
    _load_dotenv_if_present()

    API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
    api_key = (
        os.getenv("OPENAI_API_KEY", "").strip()
        or os.getenv("HF_TOKEN", "").strip()
        or os.getenv("API_KEY", "").strip()
    )
    if not api_key:
        raise RuntimeError(
            "Missing API key: set HF_TOKEN (preferred) or OPENAI_API_KEY or API_KEY"
        )

    benchmark = os.getenv("LEAD_TRIAGE_BENCHMARK", "lead_triage_env")
    env_base_url = os.getenv("LEAD_TRIAGE_ENV_BASE_URL", "http://localhost:8000")
    episodes_per_tier = int(os.getenv("EPISODES_PER_TIER", "8"))
    base_seed = int(os.getenv("BASE_SEED", "1337"))
    local_image = (
        os.getenv("LOCAL_IMAGE_NAME", "").strip()
        or os.getenv("IMAGE_NAME", "").strip()
    )

    client = OpenAI(base_url=API_BASE_URL, api_key=api_key)

    for tier_index, tier in enumerate(TIERS):
        for i in range(episodes_per_tier):
            seed = base_seed + (tier_index * 10000) + i
            default_task = f"lead_triage-{tier}-{i}"
            task_name = os.getenv("LEAD_TRIAGE_TASK", "").strip() or default_task

            # Fresh env client per episode — server closes the WS after `done`,
            # so reusing one client triggers `received 1000 (OK); then sent 1000
            # (OK)` on the next reset.
            env: Optional[LeadTriageEnv] = None
            try:
                if local_image:
                    env = await LeadTriageEnv.from_docker_image(local_image)
                else:
                    env = LeadTriageEnv(base_url=env_base_url)
                    await env.connect()

                await run_episode(
                    env,
                    client,
                    MODEL_NAME,
                    task_name,
                    benchmark,
                    tier,
                    seed,
                )
            except Exception as exc:
                print(
                    f"[DEBUG] episode failed (tier={tier} i={i}): {exc}",
                    file=sys.stderr,
                    flush=True,
                )
            finally:
                if env is not None:
                    try:
                        await env.close()
                    except Exception as close_exc:
                        print(
                            f"[DEBUG] env.close() error: {close_exc}",
                            file=sys.stderr,
                            flush=True,
                        )


def main() -> None:
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
