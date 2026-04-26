"""Prompt builder for the LLM policy.

Mirrors the message shape used by `inference.py` so that before/after
comparisons between the rule baseline, the API LLM baseline, and the trained
GRPO policy stay apples-to-apples.

The key M5 contribution here is `render_history_narrative`, which turns the
structured `contact_history` (M3d) into readable text the LLM can reason over.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Mapping, Optional

# Match inference.py exactly so the trained policy can be swapped in.
DEFAULT_ARG_BY_CHANNEL: Dict[str, str] = {
    "EMAIL": "generic",
    "CALL": "discovery",
    "FOLLOW_UP": "email:soft",
    "IGNORE": "",
}

SYSTEM_PROMPT = (
    "You are a B2B sales lead-triage policy. "
    "Each turn you receive a structured observation of one lead, the recent "
    "contact history with that lead, and the list of legal action tokens for "
    "this step. Pick the single best action token to maximise eventual "
    "conversion while respecting the per-episode contact budget and the "
    "no-spam cooldown. "
    "Reply with EXACTLY one token from `legal_actions` and nothing else. "
    "If the legal token has the form CHANNEL(argument), include the argument."
)


def _flatten_legal_tokens(
    legal_action_map: Optional[Mapping[str, Iterable[str]]],
    legal_actions: Iterable[str],
) -> List[str]:
    """Convert the structured legal-action map into the CHANNEL(arg) tokens.

    Falls back to the bare channels if no map is provided (v1 envs).
    """
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


def render_history_narrative(
    contact_history: Iterable[Mapping[str, Any]],
    history_summary: Optional[Mapping[str, Any]] = None,
    *,
    max_lines: int = 10,
) -> str:
    """Render `contact_history` as a compact, LLM-friendly narrative.

    Each line summarises one prior interaction. We cap at `max_lines` so the
    prompt size stays bounded (matches the 10-record cap on the env side).
    """
    history_list = list(contact_history)[-max_lines:]
    if not history_list and not history_summary:
        return "No prior interactions on file."

    lines: List[str] = []
    if history_summary:
        lines.append(
            "Summary: "
            f"touches={int(history_summary.get('total_touches', 0))}, "
            f"inbound={int(history_summary.get('inbound_count', 0))}, "
            f"days_since_first={int(history_summary.get('days_since_first_touch', 0))}, "
            f"days_since_last={int(history_summary.get('days_since_last_touch', 0))}, "
            f"longest_silence_days={int(history_summary.get('longest_silence_gap', 0))}, "
            f"last_inbound_sentiment={float(history_summary.get('last_inbound_sentiment', 0.0)):+.2f}"
        )
        objections = history_summary.get("raised_objections") or []
        if objections:
            lines.append("Objections raised: " + ", ".join(map(str, objections)))

    if history_list:
        lines.append("Recent interactions (oldest -> newest):")
    for record in history_list:
        day = int(record.get("day_offset", 0))
        when = f"d{day:+d}"
        channel = str(record.get("channel", "?"))
        direction = str(record.get("direction", "?"))
        outcome = str(record.get("outcome", "none"))
        topic = str(record.get("topic", "")).strip()
        sentiment = float(record.get("sentiment", 0.0))
        note = str(record.get("note", "")).strip()
        head = f"  - {when} {direction} {channel} -> {outcome} (sent {sentiment:+.2f})"
        if topic:
            head += f" topic={topic}"
        if note and note != topic:
            head += f" note={note}"
        lines.append(head)

    return "\n".join(lines)


def _observation_summary(obs: Mapping[str, Any]) -> Dict[str, Any]:
    """Pick the scalar fields the policy is allowed to see.

    Intentionally drops `trajectory` and the verbose history records (those
    are rendered separately as narrative) and any latent debug fields.
    """
    keep = (
        "company_size",
        "budget",
        "industry",
        "source",
        "engagement_score",
        "days_since_contact",
        "intent_score",
        "intent_estimate",
        "job_title",
        "contact_attempts",
        "estimated_deal_value",
        "urgency_level",
        "step_index",
        "max_steps",
        "has_prior_contact",
        "last_event",
        "task_tier",
    )
    return {k: obs.get(k) for k in keep if k in obs}


def build_prompt(
    observation: Mapping[str, Any],
    *,
    legal_action_map: Optional[Mapping[str, Iterable[str]]] = None,
) -> Dict[str, Any]:
    """Return a dict with `messages` (chat) and `legal_tokens` (resample list).

    The `messages` field is suitable for `tokenizer.apply_chat_template(...)`;
    `legal_tokens` is the post-processing whitelist used by the rollout
    collector to reject and resample malformed completions.
    """
    legal_actions = list(observation.get("legal_actions") or [])
    legal_map = legal_action_map or observation.get("legal_action_map") or None
    legal_tokens = _flatten_legal_tokens(legal_map, legal_actions)

    history_summary = observation.get("history_summary") or {}
    contact_history = observation.get("contact_history") or []
    narrative = render_history_narrative(contact_history, history_summary)
    obs_summary = _observation_summary(observation)

    user_payload = {
        "task_tier": observation.get("task_tier", "easy"),
        "step": int(observation.get("step_index", 0)),
        "horizon": int(observation.get("max_steps", 4)),
        "observation": obs_summary,
        "legal_actions": legal_tokens,
    }

    user_text = (
        "Lead snapshot:\n"
        f"{json.dumps(user_payload, separators=(',', ':'), default=str)}\n\n"
        "Contact history:\n"
        f"{narrative}\n\n"
        "Reply with exactly one token from legal_actions."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    return {"messages": messages, "legal_tokens": legal_tokens}
