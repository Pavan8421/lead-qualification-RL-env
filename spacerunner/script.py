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
import socket
import subprocess
import sys
import time
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
    algo = m0.get("algorithm", {})
    training = raw.get("training") or {}
    sampling = training.get("sampling", {})
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
        learning_rate=float(training.get("learning_rate", 5e-6)),
        kl_beta=float(algo.get("kl_beta", 0.04)),
        grad_clip=float(training.get("grad_clip", 1.0)),
        use_reference_model=bool(training.get("use_reference_model", True)),
        backend=str(training.get("backend", "auto")),
        device=str(training.get("device", "auto")),
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
    if os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true", "yes"):
        return None
    if not os.environ.get("WANDB_API_KEY"):
        print("[WARN] WANDB_API_KEY not set; skipping wandb.", file=sys.stderr)
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

    # Start the env server in-process (Spacerunner = single container).
    env_proc: Optional[subprocess.Popen] = None
    if os.environ.get("START_ENV_SERVER", "1") == "1":
        # Parse port from env_base_url; default 8000.
        from urllib.parse import urlparse

        parsed = urlparse(grpo_cfg.env_base_url)
        port = parsed.port or 8000
        env_proc = _start_env_server(port=port)

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
        if cli.model:
            policy_cfg.model_name = cli.model
        if cli.backend:
            policy_cfg.backend = cli.backend
        if cli.device:
            policy_cfg.device = cli.device
        # Auto-disable 4-bit on the HF backend (no bitsandbytes off-CUDA).
        if policy_cfg.backend in ("hf",) or (
            policy_cfg.backend == "auto" and cli.backend != "unsloth"
        ):
            try:
                import torch  # noqa: WPS433

                if not torch.cuda.is_available():
                    policy_cfg.load_in_4bit = False
            except ImportError:
                policy_cfg.load_in_4bit = False
        policy = GRPOLLMPolicy(policy_cfg).load()
        trainer = LeadTriageGRPOTrainer(grpo_cfg, policy=policy)

    run = _maybe_init_wandb(grpo_cfg, raw)
    out_dir = Path(grpo_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard writer — always on; cheap, no external account needed.
    tb_writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter  # type: ignore

        tb_dir = out_dir / "tb"
        tb_writer = SummaryWriter(log_dir=str(tb_dir))
        print(f"[TB] writing TensorBoard logs to {tb_dir}", flush=True)
    except ImportError:
        print("[TB] tensorboard not installed; skipping.", file=sys.stderr)

    try:
        zero_streak = 0
        for step in range(grpo_cfg.total_optimizer_steps):
            metrics = await trainer.step_once(step_index=step)
            print(json.dumps({"step": step, **metrics}), flush=True)
            if run is not None:
                run.log(metrics, step=step)
            if tb_writer is not None:
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        tb_writer.add_scalar(k, float(v), step)
                tb_writer.flush()

            # Watchdog: env-server death produces empty rollouts forever.
            # Detect 2 consecutive steps with no captured handles and abort
            # rather than waste hours logging zeros.
            if int(metrics.get("n_grad_samples", 0)) == 0:
                zero_streak += 1
            else:
                zero_streak = 0
            if zero_streak >= 2:
                env_alive = (
                    env_proc is None
                    or env_proc.poll() is None
                )
                print(
                    f"[FATAL] no gradient samples for {zero_streak} steps in a row; "
                    f"env_proc_alive={env_alive}. Aborting.",
                    file=sys.stderr,
                    flush=True,
                )
                return 2

            if (step + 1) % grpo_cfg.save_every_steps == 0:
                ckpt = out_dir / f"step-{step + 1:06d}"
                if hasattr(trainer.policy, "save_adapter"):
                    trainer.policy.save_adapter(str(ckpt))
                    print(f"[CKPT] saved adapter to {ckpt}", flush=True)
                    # Stream this checkpoint to the Hub immediately so it
                    # survives Space preemption / early kill.
                    try:
                        _push_outputs_to_hub(
                            str(ckpt),
                            path_in_repo=ckpt.name,
                            commit_message=f"checkpoint {ckpt.name}",
                        )
                    except Exception as e:  # noqa: BLE001
                        print(f"[HUB] mid-run upload failed: {e}", flush=True)
    finally:
        await trainer.close()
        if run is not None:
            run.finish()
        if tb_writer is not None:
            tb_writer.close()
        if env_proc is not None:
            print("[ENV] terminating server", flush=True)
            env_proc.terminate()
            try:
                env_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                env_proc.kill()

    return 0


def _wait_for_port(host: str, port: int, timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def _start_env_server(port: int = 8000) -> Optional[subprocess.Popen]:
    """Launch the FastAPI env server as a background subprocess."""
    cmd = [
        sys.executable, "-m", "uvicorn",
        "lead_triage_env.server.app:app",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--log-level", "warning",
    ]
    print(f"[ENV] starting server: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
    if not _wait_for_port("127.0.0.1", port, timeout=60.0):
        proc.terminate()
        raise RuntimeError(f"env server did not bind to port {port} within 60s")
    print(f"[ENV] server ready on 127.0.0.1:{port}", flush=True)
    return proc


def _push_outputs_to_hub(
    output_dir: str,
    path_in_repo: Optional[str] = None,
    commit_message: Optional[str] = None,
) -> None:
    """Upload `output_dir` to the HF Hub. No-op if HF_TOKEN unset.

    If `path_in_repo` is given, the folder is uploaded *as a subfolder* of the
    repo (used for per-checkpoint mid-run uploads). Otherwise the entire
    `output_dir` is mirrored at the repo root (final end-of-run upload).
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[HUB] HF_TOKEN not set; skipping upload.", flush=True)
        return
    if not Path(output_dir).exists():
        print(f"[HUB] {output_dir} does not exist; nothing to upload.", flush=True)
        return
    try:
        from huggingface_hub import HfApi  # type: ignore
    except ImportError:
        print("[HUB] huggingface_hub not installed; skipping upload.", flush=True)
        return

    repo_id = os.environ.get("OUTPUT_REPO", "pavanKumar2004/lead-triage-grpo")
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", exist_ok=True, private=True)
    kwargs: Dict[str, Any] = dict(
        folder_path=output_dir,
        repo_id=repo_id,
        repo_type="model",
    )
    if path_in_repo is not None:
        kwargs["path_in_repo"] = path_in_repo
    if commit_message is not None:
        kwargs["commit_message"] = commit_message
    api.upload_folder(**kwargs)
    dest = f"{repo_id}/{path_in_repo}" if path_in_repo else repo_id
    print(f"[HUB] uploaded {output_dir} -> {dest}", flush=True)


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
    parser.add_argument(
        "--model",
        default=None,
        help="Override m0_decisions.model.primary (e.g. Qwen/Qwen2.5-0.5B-Instruct).",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "unsloth", "hf"],
        default=None,
        help="Force a backend. `hf` skips Unsloth/4-bit (Apple Silicon / CPU).",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default=None,
    )
    parser.add_argument("--env-base-url", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    rc = asyncio.run(_async_main(args))
    # Always try to push artifacts (Spacerunner discards local files).
    raw = _load_yaml(args.config)
    output_dir = str((raw.get("training") or {}).get("output_dir", "outputs/adapters"))
    _push_outputs_to_hub(output_dir)
    sys.exit(rc)


if __name__ == "__main__":
    main()