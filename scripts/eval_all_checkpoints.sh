#!/usr/bin/env bash
# Evaluate ALL adapter checkpoints in outputs/adapters/ against the base model.
#
# For each step-NNNNNN/ folder:
#   1. Kill any running vLLM
#   2. Launch vLLM with that adapter as `trained`
#   3. Wait for vLLM ready
#   4. Run compare_models.py with guardrails OFF
#   5. Save per-checkpoint report to outputs/comparison/step-NNNNNN/
#
# Then run aggregate_eval.py to print a final summary table.
#
# Prereqs:
#   - Env server running on :8000 (LEAD_MAX_CONCURRENT_ENVS=16+)
#   - vLLM is killable by user (not started under sudo)
#   - conda env "lead" exists with vllm + tensorboard
#
# Usage:
#   ./scripts/eval_all_checkpoints.sh [episodes_per_tier]
#   ./scripts/eval_all_checkpoints.sh 4         # quick smoke
#   ./scripts/eval_all_checkpoints.sh 50        # full eval

set -euo pipefail

EPS_PER_TIER="${1:-4}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
ADAPTERS_DIR="${ADAPTERS_DIR:-outputs/adapters}"
OUT_ROOT="${OUT_ROOT:-outputs/comparison}"
API_PORT="${API_PORT:-8001}"
ENV_PORT="${ENV_PORT:-8000}"
CONDA_ENV="${CONDA_ENV:-lead}"
MAX_LORA_RANK="${MAX_LORA_RANK:-16}"
VLLM_READY_TIMEOUT="${VLLM_READY_TIMEOUT:-180}"

cd "$(dirname "$0")/.."   # repo root

# --- sanity checks ----------------------------------------------------------
if ! curl -sf "http://127.0.0.1:${ENV_PORT}/health" > /dev/null; then
  echo "[FATAL] env server not reachable on :${ENV_PORT}. Start it first:"
  echo "        LEAD_MAX_CONCURRENT_ENVS=16 uvicorn lead_triage_env.server.app:app --host 0.0.0.0 --port ${ENV_PORT}"
  exit 1
fi

if [ ! -d "$ADAPTERS_DIR" ]; then
  echo "[FATAL] $ADAPTERS_DIR not found"
  exit 1
fi

CHECKPOINTS=()
while IFS= read -r -d '' d; do
  CHECKPOINTS+=("$d")
done < <(find "$ADAPTERS_DIR" -maxdepth 1 -type d -name "step-*" -print0 | sort -z)

if [ "${#CHECKPOINTS[@]}" -eq 0 ]; then
  echo "[FATAL] no step-NNNNNN/ folders in $ADAPTERS_DIR"
  exit 1
fi

echo "[eval-all] found ${#CHECKPOINTS[@]} checkpoint(s):"
for c in "${CHECKPOINTS[@]}"; do echo "  - $c"; done
echo "[eval-all] episodes_per_tier=$EPS_PER_TIER  base=$BASE_MODEL"
echo ""

# --- helpers ----------------------------------------------------------------
kill_vllm() {
  local pids
  pids=$(lsof -ti :"$API_PORT" 2>/dev/null || true)
  if [ -n "$pids" ]; then
    echo "[eval-all] killing vLLM pids: $pids"
    kill $pids 2>/dev/null || true
    sleep 2
    # Force-kill anything still holding the port.
    pids=$(lsof -ti :"$API_PORT" 2>/dev/null || true)
    if [ -n "$pids" ]; then kill -9 $pids 2>/dev/null || true; fi
  fi
}

wait_vllm_ready() {
  local elapsed=0
  while [ "$elapsed" -lt "$VLLM_READY_TIMEOUT" ]; do
    if curl -sf "http://127.0.0.1:${API_PORT}/v1/models" > /dev/null 2>&1; then
      # Also check both models are listed
      local models
      models=$(curl -s "http://127.0.0.1:${API_PORT}/v1/models" \
        | python -c "import sys,json; d=json.load(sys.stdin)['data']; print(' '.join(m['id'] for m in d))" 2>/dev/null || echo "")
      if echo "$models" | grep -q "trained" && echo "$models" | grep -q "Qwen"; then
        echo "[eval-all] vLLM ready (${elapsed}s). Models: $models"
        return 0
      fi
    fi
    sleep 5
    elapsed=$((elapsed + 5))
  done
  echo "[eval-all] vLLM did not become ready within ${VLLM_READY_TIMEOUT}s"
  return 1
}

# --- main loop --------------------------------------------------------------
mkdir -p "$OUT_ROOT"
SUMMARY_PATHS=()

for ckpt in "${CHECKPOINTS[@]}"; do
  step_name=$(basename "$ckpt")
  out_dir="${OUT_ROOT}/${step_name}"
  echo ""
  echo "=========================================================================="
  echo "[eval-all] === $step_name ==="
  echo "=========================================================================="

  kill_vllm
  echo "[eval-all] launching vLLM with adapter: $ckpt"

  # Launch vLLM in background; capture log.
  log_file="${OUT_ROOT}/vllm_${step_name}.log"
  nohup conda run -n "$CONDA_ENV" python -m vllm.entrypoints.openai.api_server \
      --model "$BASE_MODEL" \
      --enable-lora \
      --lora-modules "trained=./${ckpt}" \
      --max-lora-rank "$MAX_LORA_RANK" \
      --port "$API_PORT" \
      > "$log_file" 2>&1 &
  vllm_pid=$!
  echo "[eval-all] vLLM pid=$vllm_pid  log=$log_file"

  if ! wait_vllm_ready; then
    echo "[eval-all] SKIPPING $step_name — vLLM never became ready. Check $log_file"
    kill_vllm
    continue
  fi

  echo "[eval-all] running compare_models.py (guardrails OFF, eps=$EPS_PER_TIER)"
  mkdir -p "$out_dir"
  if INFERENCE_DISABLE_GUARDRAILS=1 conda run -n "$CONDA_ENV" python scripts/compare_models.py \
        --base-model "$BASE_MODEL" \
        --trained-model "trained" \
        --episodes-per-tier "$EPS_PER_TIER" \
        --api-base-url "http://127.0.0.1:${API_PORT}/v1" \
        --out-dir "$out_dir" \
        > "${out_dir}/report.txt" 2>&1; then
    echo "[eval-all] OK -> ${out_dir}/report.txt"
    SUMMARY_PATHS+=("$out_dir")
  else
    echo "[eval-all] FAILED for $step_name. See ${out_dir}/report.txt"
  fi
done

# --- cleanup + summary ------------------------------------------------------
kill_vllm
echo ""
echo "=========================================================================="
echo "[eval-all] done. Aggregating results..."
echo "=========================================================================="
echo ""

conda run -n "$CONDA_ENV" python scripts/aggregate_eval.py \
  --comparison-root "$OUT_ROOT" \
  --base-model "$BASE_MODEL"
