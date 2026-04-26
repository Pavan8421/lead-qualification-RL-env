#!/usr/bin/env bash
# scripts/start_training.sh — one-command start point.
#
# 1. Boots the LeadTriageEnv FastAPI server in the background (env_version=v2).
# 2. Waits until /state responds.
# 3. Launches train.py with the LLM policy + real GRPO gradient updates.
# 4. Cleans up the server on exit.
#
# Override anything via env vars, e.g.:
#   STEPS=200 TIER_MIX=easy CONDA_ENV=lead ./scripts/start_training.sh
#   POLICY=stub NO_WANDB=1 ./scripts/start_training.sh   # CPU smoke
#
# Requires (for POLICY=llm): CUDA + `pip install -r requirements-train.txt`.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONDA_ENV="${CONDA_ENV:-lead}"
POLICY="${POLICY:-llm}"                     # llm | stub
STEPS="${STEPS:-50}"
ENV_PORT="${ENV_PORT:-8000}"
ENV_HOST="${ENV_HOST:-127.0.0.1}"
ENV_VERSION="${LEAD_ENV_VERSION:-v2}"
NO_WANDB="${NO_WANDB:-}"
CONFIG="${CONFIG:-configs/training.yaml}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
# Backend / device / model overrides — empty means "use YAML default".
BACKEND="${BACKEND:-}"          # auto | unsloth | hf
DEVICE="${DEVICE:-}"            # auto | cuda | mps | cpu
MODEL="${MODEL:-}"              # e.g. Qwen/Qwen2.5-0.5B-Instruct on Mac

# Mac convenience: when running natively on Apple Silicon, pick the small
# model + HF backend + MPS device automatically (override with explicit env).
if [[ "$(uname -s)-$(uname -m)" == "Darwin-arm64" && "${POLICY}" == "llm" ]]; then
  : "${BACKEND:=hf}"
  : "${DEVICE:=mps}"
  : "${MODEL:=Qwen/Qwen2.5-0.5B-Instruct}"
fi

mkdir -p "$LOG_DIR" outputs/adapters

PY="conda run -n ${CONDA_ENV} --no-capture-output python"
ENV_LOG="${LOG_DIR}/env-server.log"
TRAIN_LOG="${LOG_DIR}/train.log"

echo "[start] launching env server on ${ENV_HOST}:${ENV_PORT} (env_version=${ENV_VERSION})"
LEAD_ENV_VERSION="${ENV_VERSION}" \
  ${PY} -m uvicorn lead_triage_env.server.app:app \
    --host "${ENV_HOST}" --port "${ENV_PORT}" \
    > "${ENV_LOG}" 2>&1 &
ENV_PID=$!
trap 'echo "[start] stopping env server (pid ${ENV_PID})"; kill ${ENV_PID} 2>/dev/null || true' EXIT

# Poll /state until ready (max 30s).
echo "[start] waiting for env to come up..."
for _ in $(seq 1 60); do
  if curl -sf "http://${ENV_HOST}:${ENV_PORT}/state" > /dev/null 2>&1; then
    echo "[start] env is up"
    break
  fi
  sleep 0.5
done

if ! curl -sf "http://${ENV_HOST}:${ENV_PORT}/state" > /dev/null 2>&1; then
  echo "[start] ERROR: env did not become healthy. See ${ENV_LOG}" >&2
  tail -20 "${ENV_LOG}" >&2 || true
  exit 1
fi

WANDB_FLAG=""
if [[ -n "${NO_WANDB}" ]]; then
  WANDB_FLAG="--no-wandb"
fi

EXTRA_FLAGS=()
[[ -n "${BACKEND}" ]] && EXTRA_FLAGS+=(--backend "${BACKEND}")
[[ -n "${DEVICE}"  ]] && EXTRA_FLAGS+=(--device "${DEVICE}")
[[ -n "${MODEL}"   ]] && EXTRA_FLAGS+=(--model "${MODEL}")

echo "[start] launching trainer (policy=${POLICY}, steps=${STEPS}, backend=${BACKEND:-yaml}, device=${DEVICE:-yaml}, model=${MODEL:-yaml})"
${PY} train.py \
  --config "${CONFIG}" \
  --policy "${POLICY}" \
  --steps "${STEPS}" \
  --env-base-url "http://${ENV_HOST}:${ENV_PORT}" \
  ${WANDB_FLAG} \
  "${EXTRA_FLAGS[@]}" \
  2>&1 | tee "${TRAIN_LOG}"
