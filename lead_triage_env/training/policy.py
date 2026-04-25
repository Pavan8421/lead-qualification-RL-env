"""LLM policy wrapper for GRPO training (roadmap §M6 + real gradient step).

Wraps Unsloth's `FastLanguageModel` so we get 4-bit loading + Triton kernels +
LoRA in a couple of lines. The class now does three things:

  1. Sample completions for the rollout collector (`sample_text` /
     `sample_with_ids`).
  2. Re-score those completions to get per-token logprobs against the
     CURRENT model and against a frozen REFERENCE copy of the base model
     (`compute_logprobs`, `compute_kl_to_ref`).
  3. Apply one GRPO gradient update per group of completions
     (`grpo_update`).

This is a from-scratch GRPO step (advantage-weighted policy gradient with a
KL-to-reference penalty), not TRL's `GRPOTrainer.training_step`. The reason:
TRL owns generation internally, but our completions are env-conditioned —
each prompt depends on the previous env step's observation. Plugging that
into TRL's `Dataset` machinery costs more than the ~120 LOC update loop
below.

Heavy deps (`unsloth`, `transformers`, `peft`, `torch`) are imported lazily
so this file is importable on CPU-only machines for static analysis.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


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

    # Optimizer / GRPO knobs (read by trainer_grpo.py).
    learning_rate: float = 5.0e-6
    kl_beta: float = 0.04
    grad_clip: float = 1.0
    # If True, holds a frozen copy of the base model on the same device as
    # the reference policy for KL computation. Costs ~1× model memory; turn
    # off (`kl_beta=0`) on tight VRAM budgets.
    use_reference_model: bool = True


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
        self._ref_model: Any = None
        self._optimizer: Any = None
        # Per-sample cache of (prompt_ids, completion_ids) keyed by an int handle.
        # Filled by `sample_with_ids`, drained by `grpo_update`.
        self._sample_cache: Dict[int, Tuple[Any, Any]] = {}
        self._next_handle = 0

    # ------------------------------------------------------------ load ----

    def load(self) -> "GRPOLLMPolicy":
        """Load Unsloth 4-bit base + apply LoRA adapter (PEFT) + reference model + optimizer."""
        try:
            from unsloth import FastLanguageModel  # type: ignore
        except ImportError as exc:  # pragma: no cover - env-specific
            raise RuntimeError(
                "unsloth is required for training. Install requirements-train.txt "
                "in a CUDA-enabled venv."
            ) from exc
        import torch  # noqa: WPS433

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

        # Frozen reference copy (no LoRA, base 4-bit weights only) for KL.
        if self.config.use_reference_model and self.config.kl_beta > 0:
            ref_model, _ = FastLanguageModel.from_pretrained(
                model_name=self.config.model_name,
                max_seq_length=self.config.max_seq_length,
                dtype=dtype,
                load_in_4bit=self.config.load_in_4bit,
            )
            ref_model.eval()
            for p in ref_model.parameters():
                p.requires_grad_(False)
            self._ref_model = ref_model
        else:
            self._ref_model = None

        # Optimizer over LoRA params only (PEFT freezes the base).
        trainable = [p for p in self._model.parameters() if p.requires_grad]
        self._optimizer = torch.optim.AdamW(
            trainable, lr=self.config.learning_rate, betas=(0.9, 0.95)
        )
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

    def sample_with_ids(
        self, messages: List[Dict[str, str]]
    ) -> Tuple[str, int]:
        """Sample one completion and cache the (prompt_ids, completion_ids).

        Returns `(decoded_text, handle)`. The handle is later passed to
        `grpo_update(...)` so the trainer can reuse the exact tokenization
        that produced this rollout (no re-tokenization drift).
        """
        try:
            from unsloth import FastLanguageModel  # type: ignore

            FastLanguageModel.for_inference(self.model)
        except ImportError:  # pragma: no cover
            pass
        import torch  # noqa: WPS433

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
        with torch.no_grad():
            out = self.model.generate(prompt_ids, **gen_kwargs)
        completion_ids = out[0, prompt_ids.shape[-1]:]
        text = tokenizer.decode(completion_ids, skip_special_tokens=True)

        handle = self._next_handle
        self._next_handle += 1
        # Store on CPU to free GPU mem; we move back during grpo_update.
        self._sample_cache[handle] = (
            prompt_ids.detach().to("cpu"),
            completion_ids.detach().to("cpu"),
        )
        return text, handle

    def sample_text(self, messages: List[Dict[str, str]]) -> str:
        """Backward-compatible wrapper used by the rollout collector when no
        gradient step is needed (eval / inspection). For training use
        `sample_with_ids` so the prompt/completion ids are captured."""
        text, _ = self.sample_with_ids(messages)
        return text

    def as_async_policy_fn(
        self, *, capture_ids: bool = False
    ) -> Callable[..., Any]:
        """Adapt sampling to the async `PolicyFn` shape used by rollout.py.

        When `capture_ids=True`, the policy function returns `(text, handle)`
        and the rollout collector stores the handle on each `StepRecord`.
        """

        async def _policy_fn_text(
            messages: List[Dict[str, str]], legal_tokens: List[str]
        ) -> str:
            del legal_tokens
            return await asyncio.to_thread(self.sample_text, messages)

        async def _policy_fn_with_ids(
            messages: List[Dict[str, str]], legal_tokens: List[str]
        ) -> Tuple[str, int]:
            del legal_tokens
            return await asyncio.to_thread(self.sample_with_ids, messages)

        return _policy_fn_with_ids if capture_ids else _policy_fn_text

    # ----------------------------------------------------- logprob / KL ----

    def _completion_logprobs(
        self,
        model: Any,
        prompt_ids: Any,
        completion_ids: Any,
    ) -> Any:
        """Sum of per-token logprobs of `completion_ids` given `prompt_ids`."""
        import torch  # noqa: WPS433
        import torch.nn.functional as F  # noqa: WPS433

        device = model.device
        prompt = prompt_ids.to(device).reshape(1, -1)
        completion = completion_ids.to(device).reshape(1, -1)
        if completion.numel() == 0:
            return torch.zeros((), device=device)
        full = torch.cat([prompt, completion], dim=1)
        out = model(input_ids=full)
        logits = out.logits  # (1, T, V)
        # Predict tokens at positions [prompt_len .. prompt_len+comp_len-1];
        # the logit that predicts position i is at index i-1.
        start = prompt.shape[1] - 1
        end = full.shape[1] - 1
        logits_slice = logits[:, start:end, :]
        log_probs = F.log_softmax(logits_slice, dim=-1)
        token_logp = log_probs.gather(-1, completion.unsqueeze(-1)).squeeze(-1)
        return token_logp.sum()

    def compute_logprobs(self, handle: int) -> Any:
        """Logprob of the cached completion under the CURRENT (LoRA) model."""
        prompt_ids, completion_ids = self._sample_cache[handle]
        return self._completion_logprobs(self.model, prompt_ids, completion_ids)

    def compute_kl_to_ref(self, handle: int) -> Any:
        """Approximate KL(policy || ref) on the sampled completion.

        We use the per-sequence "logprob difference" estimator
        (E_{x~pi}[log pi(x) - log pi_ref(x)]); cheap, unbiased on-policy,
        standard in GRPO / RLHF references.
        """
        import torch  # noqa: WPS433

        if self._ref_model is None:
            return torch.zeros((), device=self.model.device)
        prompt_ids, completion_ids = self._sample_cache[handle]
        with torch.no_grad():
            ref_logp = self._completion_logprobs(
                self._ref_model, prompt_ids, completion_ids
            )
        cur_logp = self.compute_logprobs(handle)
        return (cur_logp - ref_logp).detach() * 0.0 + (cur_logp - ref_logp)

    def release_handles(self, handles: Sequence[int]) -> None:
        for h in handles:
            self._sample_cache.pop(h, None)

    # ------------------------------------------------------ grad step -----

    def grpo_update(
        self,
        handles: Sequence[int],
        advantages: Sequence[float],
    ) -> Dict[str, float]:
        """One GRPO optimizer step over a batch of (handle, advantage) pairs.

        Loss per sample:
            L_i = -A_i * logπ(c_i | p_i) + β * KL(π || π_ref)(c_i | p_i)
        Aggregated:
            L = mean_i(L_i)

        Returns a dict of scalar metrics for W&B.
        """
        if not handles:
            return {"loss": 0.0, "policy_loss": 0.0, "kl": 0.0, "n": 0.0}
        import torch  # noqa: WPS433

        assert len(handles) == len(advantages), "handles/advantages length mismatch"

        self._model.train()
        self._optimizer.zero_grad(set_to_none=True)

        device = self._model.device
        adv_tensor = torch.tensor(
            list(advantages), device=device, dtype=torch.float32
        )

        per_sample_pg: List[Any] = []
        per_sample_kl: List[Any] = []
        for h in handles:
            cur_logp = self.compute_logprobs(h)
            kl = self.compute_kl_to_ref(h) if self.config.kl_beta > 0 else torch.zeros((), device=device)
            per_sample_pg.append(cur_logp)
            per_sample_kl.append(kl)

        logp_stack = torch.stack(per_sample_pg).to(torch.float32)
        kl_stack = torch.stack(per_sample_kl).to(torch.float32)

        policy_loss = -(adv_tensor * logp_stack).mean()
        kl_loss = kl_stack.mean()
        loss = policy_loss + self.config.kl_beta * kl_loss

        loss.backward()
        if self.config.grad_clip and self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self._model.parameters() if p.requires_grad],
                self.config.grad_clip,
            )
        self._optimizer.step()
        self.release_handles(handles)

        return {
            "loss": float(loss.detach().item()),
            "policy_loss": float(policy_loss.detach().item()),
            "kl": float(kl_loss.detach().item()),
            "mean_logprob": float(logp_stack.mean().detach().item()),
            "n": float(len(handles)),
        }

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
