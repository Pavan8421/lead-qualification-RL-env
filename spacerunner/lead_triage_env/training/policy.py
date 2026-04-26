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

    # Backend selection. "auto" picks unsloth if importable + CUDA, else HF
    # transformers on the best available device (cuda > mps > cpu).
    backend: str = "auto"          # "auto" | "unsloth" | "hf"
    device: str = "auto"           # "auto" | "cuda" | "mps" | "cpu"

    # GRPO grad-accumulation chunk size. With B=12, G=16 we get ~192 handles
    # per step; building one autograd graph over all of them OOMs small
    # GPUs / MPS. Process this many handles per micro-batch and accumulate
    # gradients before stepping. Set <=0 to disable (single big batch).
    grpo_micro_batch_size: int = 16


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
        # Serialize all GPU-touching code. The rollout collector calls
        # `sample_with_ids` from up to G concurrent threads via
        # `asyncio.to_thread`; HF/Unsloth `model.generate` is NOT thread-safe
        # (rotary cache extension, KV cache allocation, Unsloth's
        # `for_inference` toggle all mutate shared state). Concurrent calls
        # corrupt position_ids -> CUDA index-out-of-bounds -> inf/nan logits.
        import threading as _threading  # noqa: WPS433

        self._gpu_lock = _threading.Lock()

    # ------------------------------------------------------------ load ----

    def _resolve_backend(self) -> str:
        if self.config.backend != "auto":
            return self.config.backend
        try:
            import torch  # noqa: WPS433

            cuda_ok = torch.cuda.is_available()
        except ImportError:
            cuda_ok = False
        if not cuda_ok:
            return "hf"
        try:
            import unsloth  # type: ignore  # noqa: F401
        except ImportError:
            return "hf"
        return "unsloth"

    def _resolve_device(self) -> str:
        if self.config.device != "auto":
            return self.config.device
        import torch  # noqa: WPS433

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load(self) -> "GRPOLLMPolicy":
        """Load model + LoRA + optional reference + AdamW.

        Picks Unsloth (CUDA + 4-bit) when available, otherwise falls back to
        plain HF transformers + PEFT on the best local device (MPS on Apple
        Silicon, CPU on everything else without a GPU).
        """
        backend = self._resolve_backend()
        if backend == "unsloth":
            self._load_unsloth()
        else:
            self._load_hf()
        self._build_optimizer()
        return self

    # --- backend: unsloth (CUDA + 4-bit) -----------------------------------

    def _load_unsloth(self) -> None:
        from unsloth import FastLanguageModel  # type: ignore

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

    # --- backend: HF transformers + PEFT (CUDA / MPS / CPU) ----------------

    def _load_hf(self) -> None:
        try:
            import torch  # noqa: WPS433
            from peft import LoraConfig, get_peft_model  # type: ignore
            from transformers import (  # type: ignore
                AutoModelForCausalLM,
                AutoTokenizer,
            )
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "transformers + peft are required for the HF backend. "
                "Install requirements-train-cpu.txt (Apple Silicon / CPU) or "
                "requirements-train.txt (CUDA)."
            ) from exc

        device = self._resolve_device()
        # MPS doesn't support bfloat16 on all ops yet -> fall back to float16.
        dtype_name = self.config.dtype
        if device == "mps" and dtype_name.lower() in ("bfloat16", "bf16"):
            dtype_name = "float16"
        if device == "cpu" and dtype_name.lower() in ("float16", "fp16"):
            dtype_name = "float32"
        dtype = _resolve_dtype(dtype_name)

        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=dtype,
        ).to(device)
        base.config.pad_token_id = tokenizer.pad_token_id

        lora_cfg = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self._lora_target_modules(),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base, lora_cfg)
        # Make sure LoRA params are trainable & in fp32 (numerical stability).
        for n, p in model.named_parameters():
            if "lora_" in n:
                p.requires_grad_(True)
                if p.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                    p.data = p.data.to(torch.float32)
        self._model = model
        self._tokenizer = tokenizer

        if self.config.use_reference_model and self.config.kl_beta > 0:
            ref = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=dtype,
            ).to(device)
            ref.config.pad_token_id = tokenizer.pad_token_id
            ref.eval()
            for p in ref.parameters():
                p.requires_grad_(False)
            self._ref_model = ref
        else:
            self._ref_model = None

    def _build_optimizer(self) -> None:
        import torch  # noqa: WPS433

        trainable = [p for p in self._model.parameters() if p.requires_grad]
        self._optimizer = torch.optim.AdamW(
            trainable, lr=self.config.learning_rate, betas=(0.9, 0.95)
        )

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
        # Serialize: HF/Unsloth generation is not thread-safe under the
        # async-fanout rollout collector (see __init__).
        with self._gpu_lock:
            return self._sample_with_ids_locked(messages)

    def _sample_with_ids_locked(
        self, messages: List[Dict[str, str]]
    ) -> Tuple[str, int]:
        try:
            from unsloth import FastLanguageModel  # type: ignore

            FastLanguageModel.for_inference(self.model)
        except ImportError:
            # HF backend / non-Unsloth: just toggle eval mode.
            try:
                self.model.eval()
            except AttributeError:
                pass
        import torch  # noqa: WPS433

        tokenizer = self.tokenizer
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        if hasattr(encoded, "input_ids"):
            prompt_ids = encoded["input_ids"]
            attention_mask = encoded.get("attention_mask")
        else:
            prompt_ids = encoded
            attention_mask = None
        prompt_ids = prompt_ids.to(self.model.device)
        if attention_mask is None:
            # Build a trivial all-ones mask. Without this, Qwen (pad==eos)
            # silently computes wrong position_ids, which crashes the rotary
            # embedding gather with an index-out-of-bounds CUDA assert.
            attention_mask = torch.ones_like(prompt_ids)
        else:
            attention_mask = attention_mask.to(self.model.device)

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.temperature > 0.0,
            "temperature": max(self.config.temperature, 1e-5),
            "top_p": self.config.top_p,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            "attention_mask": attention_mask,
            # Re-normalize after temperature / top_p warpers so any inf/-inf
            # leaking from bf16 lm_head doesn't poison the multinomial draw.
            "renormalize_logits": True,
        }
        try:
            with torch.no_grad():
                out = self.model.generate(prompt_ids, **gen_kwargs)
        except RuntimeError as exc:
            # `probability tensor contains either inf, nan or element < 0` — fall
            # back to greedy decoding for this one sample so a single bad prompt
            # cannot kill a multi-hour training run.
            msg = str(exc).lower()
            if ("inf" in msg or "nan" in msg) and "probability" in msg:
                fallback = dict(gen_kwargs)
                fallback.update(
                    {"do_sample": False, "temperature": 1.0, "top_p": 1.0}
                )
                with torch.no_grad():
                    out = self.model.generate(prompt_ids, **fallback)
            else:
                raise
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
        # Explicit attention mask. Without it, Qwen2 (pad==eos) silently
        # computes wrong position_ids and the rotary embedding gather throws
        # `index out of bounds` on CUDA. `use_cache=False` skips KV-cache
        # allocation — we don't need it for a single forward pass.
        attn_mask = torch.ones_like(full)
        out = model(input_ids=full, attention_mask=attn_mask, use_cache=False)
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
        n_total = len(handles)
        chunk = int(self.config.grpo_micro_batch_size or 0)
        if chunk <= 0:
            chunk = n_total

        sum_loss = 0.0
        sum_policy_loss = 0.0
        sum_kl = 0.0
        sum_logp = 0.0

        for start in range(0, n_total, chunk):
            end = min(start + chunk, n_total)
            chunk_handles = list(handles[start:end])
            chunk_advs = list(advantages[start:end])
            adv_tensor = torch.tensor(chunk_advs, device=device, dtype=torch.float32)

            per_sample_pg: List[Any] = []
            per_sample_kl: List[Any] = []
            for h in chunk_handles:
                cur_logp = self.compute_logprobs(h)
                if self.config.kl_beta > 0 and self._ref_model is not None:
                    prompt_ids, completion_ids = self._sample_cache[h]
                    with torch.no_grad():
                        ref_logp = self._completion_logprobs(
                            self._ref_model, prompt_ids, completion_ids
                        )
                    kl = cur_logp - ref_logp
                else:
                    kl = torch.zeros((), device=device)
                per_sample_pg.append(cur_logp)
                per_sample_kl.append(kl)

            logp_stack = torch.stack(per_sample_pg).to(torch.float32)
            kl_stack = torch.stack(per_sample_kl).to(torch.float32)

            scale = float(end - start) / float(n_total)
            policy_loss = -(adv_tensor * logp_stack).mean() * scale
            kl_loss = kl_stack.mean() * scale
            loss = policy_loss + self.config.kl_beta * kl_loss

            loss.backward()

            sum_loss += float(loss.detach().item())
            sum_policy_loss += float(policy_loss.detach().item())
            sum_kl += float(kl_loss.detach().item())
            sum_logp += float(logp_stack.sum().detach().item())

            del per_sample_pg, per_sample_kl, logp_stack, kl_stack, loss

        if self.config.grad_clip and self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self._model.parameters() if p.requires_grad],
                self.config.grad_clip,
            )
        self._optimizer.step()
        self.release_handles(handles)

        return {
            "loss": sum_loss,
            "policy_loss": sum_policy_loss,
            "kl": sum_kl,
            "mean_logprob": sum_logp / max(1, n_total),
            "n": float(n_total),
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
