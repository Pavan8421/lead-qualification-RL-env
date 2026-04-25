"""LLM policy wrapper for GRPO training (roadmap §M6).

Wraps Unsloth's `FastLanguageModel` so we get 4-bit loading + Triton kernels +
LoRA in a couple of lines. Greedy decode is fine for our action surface
(tokens are short — `EMAIL(case_study)` is at most 8 tokens).

Heavy deps (`unsloth`, `transformers`, `peft`) are imported lazily so this
file is importable on CPU-only machines for static analysis / tests.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class PolicyConfig:
    """Policy hyperparameters loaded from `configs/training.yaml::m0_decisions`."""

    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    load_in_4bit: bool = True
    dtype: str = "bfloat16"
    max_seq_length: int = 2048
    max_new_tokens: int = 16
    temperature: float = 0.7  # raised from 0 for GRPO sampling diversity
    top_p: float = 0.9

    lora_rank: int = 16
    lora_alpha: int = 32
    lora_target: str = "all-linear"
    lora_dropout: float = 0.0


_DTYPE_MAP = {
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
    "float16": "float16",
    "fp16": "float16",
    "float32": "float32",
    "fp32": "float32",
}


def _resolve_dtype(name: str) -> Any:
    """Lazy import torch only when actually loading a model."""
    import torch  # noqa: WPS433 (intentional lazy import)

    canonical = _DTYPE_MAP.get(name.lower(), "bfloat16")
    return getattr(torch, canonical)


class GRPOLLMPolicy:
    """Thin wrapper around an Unsloth/HF model for rollout sampling.

    The trainer (TRL GRPOTrainer) talks to the underlying model directly; the
    rollout collector (M5) talks through `as_async_policy_fn()`.
    """

    def __init__(self, config: Optional[PolicyConfig] = None) -> None:
        self.config = config or PolicyConfig()
        self._model: Any = None
        self._tokenizer: Any = None

    # ------------------------------------------------------------ load ----

    def load(self) -> "GRPOLLMPolicy":
        """Load Unsloth 4-bit base + apply LoRA adapter (PEFT)."""
        try:
            from unsloth import FastLanguageModel  # type: ignore
        except ImportError as exc:  # pragma: no cover - env-specific
            raise RuntimeError(
                "unsloth is required for training. Install requirements-train.txt "
                "in a CUDA-enabled venv."
            ) from exc

        dtype = _resolve_dtype(self.config.dtype)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            dtype=dtype,
            load_in_4bit=self.config.load_in_4bit,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self._lora_target_modules(),
            bias="none",
            use_gradient_checkpointing="unsloth",
        )
        self._model = model
        self._tokenizer = tokenizer
        return self

    def _lora_target_modules(self) -> List[str]:
        # Unsloth accepts the literal "all-linear" via PEFT >= 0.10.
        if self.config.lora_target == "all-linear":
            return [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        return [t.strip() for t in self.config.lora_target.split(",") if t.strip()]

    @property
    def model(self) -> Any:
        if self._model is None:
            raise RuntimeError("Call .load() first.")
        return self._model

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            raise RuntimeError("Call .load() first.")
        return self._tokenizer

    # ---------------------------------------------------------- sample ----

    def sample_text(self, messages: List[Dict[str, str]]) -> str:
        """Single greedy-ish completion for one prompt (sync)."""
        try:
            from unsloth import FastLanguageModel  # type: ignore

            FastLanguageModel.for_inference(self.model)
        except ImportError:  # pragma: no cover
            pass

        tokenizer = self.tokenizer
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.temperature > 0.0,
            "temperature": max(self.config.temperature, 1e-5),
            "top_p": self.config.top_p,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        }
        out = self.model.generate(prompt_ids, **gen_kwargs)
        new_tokens = out[0, prompt_ids.shape[-1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    def as_async_policy_fn(self) -> Callable[..., Any]:
        """Adapt `sample_text` to the async `PolicyFn` shape used by rollout.py.

        Generation is inherently blocking; we offload to a thread so the
        async rollout loop is not stalled.
        """

        async def _policy_fn(
            messages: List[Dict[str, str]], legal_tokens: List[str]
        ) -> str:
            del legal_tokens  # mask enforcement happens in rollout.parse_action_token
            return await asyncio.to_thread(self.sample_text, messages)

        return _policy_fn

    # ------------------------------------------------------------ save ----

    def save_adapter(self, output_dir: str) -> None:
        """Persist LoRA adapter only — never naively merge from 4-bit (§16)."""
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)


# ---------- helpers for non-LLM smoke runs (no GPU required) -------------

_TOKEN_HINT_RE = re.compile(r"\b(EMAIL|CALL|FOLLOW_UP|IGNORE)\b")


def make_stub_policy_fn() -> Callable[..., Any]:
    """Return an async policy fn that picks the safest legal token.

    Useful for unit-testing the rollout collector without loading a model.
    """

    async def _stub(
        messages: List[Dict[str, str]], legal_tokens: List[str]
    ) -> str:
        for preferred in (
            "EMAIL(value_prop)",
            "EMAIL(generic)",
            "CALL(discovery)",
            "FOLLOW_UP(email:soft)",
            "IGNORE",
        ):
            if preferred in legal_tokens:
                return preferred
        return legal_tokens[0] if legal_tokens else "IGNORE"

    return _stub
