"""Training entrypoint (roadmap §M6).

Usage:
    python train.py --config configs/training.yaml --steps 1
    python train.py --config configs/training.yaml --steps 50 --policy stub

`--policy stub` skips loading the LLM and uses a deterministic policy from
`policy.make_stub_policy_fn()` — useful for CI smoke tests on CPU.

Heavy ML deps (torch/unsloth/trl/wandb) are imported only when actually
needed, so `python train.py --help` works even without them installed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from lead_triage_env.training.policy import (
    GRPOLLMPolicy,
    PolicyConfig,
    make_stub_policy_fn,
)
from lead_triage_env.training.trainer_grpo import GRPOConfig, LeadTriageGRPOTrainer


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _build_policy_config(raw: Dict[str, Any]) -> PolicyConfig:
    m0 = raw.get("m0_decisions", {})
    model_block = m0.get("model", {})
    eff_block = m0.get("efficiency", {})
    sampling = (raw.get("training") or {}).get("sampling", {})
    return PolicyConfig(
        model_name=model_block.get("primary", PolicyConfig.model_name),
        load_in_4bit=bool(model_block.get("load_in_4bit", True)),
        dtype=str(model_block.get("dtype", "bfloat16")),
        max_new_tokens=int(sampling.get("max_new_tokens", 16)),
        temperature=float(sampling.get("temperature", 0.7)),
        top_p=float(sampling.get("top_p", 0.9)),
        lora_rank=int(eff_block.get("lora_rank", 16)),
        lora_alpha=int(eff_block.get("lora_alpha", 32)),
        lora_target=str(eff_block.get("lora_target", "all-linear")),
    )


def _build_grpo_config(raw: Dict[str, Any], cli: argparse.Namespace) -> GRPOConfig:
    m0 = raw.get("m0_decisions", {})
    training = raw.get("training") or {}
    algo = m0.get("algorithm", {})
    seeds = m0.get("seeds", {})
    base_seed = (
        seeds.get("train_range", [1337])[0]
        if isinstance(seeds.get("train_range"), list)
        else 1337
    )
    return GRPOConfig(
        output_dir=str(training.get("output_dir", "outputs/adapters")),
        learning_rate=float(training.get("learning_rate", 5e-6)),
        num_generations=int(algo.get("num_generations", 4)),
        kl_beta=float(algo.get("kl_beta", 0.04)),
        episodes_per_step=int(training.get("episodes_per_step", 8)),
        total_optimizer_steps=int(cli.steps or training.get("total_optimizer_steps", 50)),
        lambda_terminal_grader=float(training.get("lambda_terminal_grader", 2.0)),
        save_every_steps=int(training.get("save_every_steps", 25)),
        eval_every_steps=int(training.get("eval_every_steps", 25)),
        seed=int(training.get("seed", base_seed)),
        tier_mix=tuple(training.get("tier_mix", ["easy"])),
        env_base_url=str(
            cli.env_base_url
            or os.environ.get("LEAD_TRIAGE_ENV_BASE_URL", "http://localhost:8000")
        ),
        env_concurrency=int(training.get("env_concurrency", 4)),
        log_to_wandb=bool(training.get("log_to_wandb", True))
        and not cli.no_wandb,
        wandb_project=str(
            (m0.get("logging") or {}).get("project", "lead-triage-rl")
        ),
    )


def _maybe_init_wandb(grpo_cfg: GRPOConfig, raw: Dict[str, Any]) -> Optional[Any]:
    if not grpo_cfg.log_to_wandb:
        return None
    try:
        import wandb  # type: ignore
    except ImportError:
        print("[WARN] wandb not installed; continuing without logging.", file=sys.stderr)
        return None
    run = wandb.init(project=grpo_cfg.wandb_project, config=raw)
    return run


async def _async_main(cli: argparse.Namespace) -> int:
    raw = _load_yaml(cli.config)
    grpo_cfg = _build_grpo_config(raw, cli)

    if cli.policy == "stub":
        # Use stub policy by monkey-injecting it onto a bare policy object —
        # no model load, no torch import.
        class _StubPolicy:
            def as_async_policy_fn(self):  # noqa: D401
                return make_stub_policy_fn()

            def save_adapter(self, output_dir: str) -> None:  # noqa: D401
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                Path(output_dir, "STUB.txt").write_text("stub policy", encoding="utf-8")

        trainer = LeadTriageGRPOTrainer(grpo_cfg, policy=_StubPolicy())  # type: ignore[arg-type]
    else:
        policy_cfg = _build_policy_config(raw)
        policy = GRPOLLMPolicy(policy_cfg).load()
        trainer = LeadTriageGRPOTrainer(grpo_cfg, policy=policy)

    run = _maybe_init_wandb(grpo_cfg, raw)
    out_dir = Path(grpo_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        for step in range(grpo_cfg.total_optimizer_steps):
            metrics = await trainer.step_once(step_index=step)
            print(json.dumps({"step": step, **metrics}), flush=True)
            if run is not None:
                run.log(metrics, step=step)

            if (step + 1) % grpo_cfg.save_every_steps == 0:
                ckpt = out_dir / f"step-{step + 1:06d}"
                if hasattr(trainer.policy, "save_adapter"):
                    trainer.policy.save_adapter(str(ckpt))
                    print(f"[CKPT] saved adapter to {ckpt}", flush=True)
    finally:
        await trainer.close()
        if run is not None:
            run.finish()

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Lead-triage GRPO trainer.")
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--steps", type=int, default=None, help="Override total_optimizer_steps.")
    parser.add_argument(
        "--policy",
        choices=["llm", "stub"],
        default="llm",
        help="`stub` skips model load — for CPU smoke tests.",
    )
    parser.add_argument("--env-base-url", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    sys.exit(asyncio.run(_async_main(args)))


if __name__ == "__main__":
    main()
