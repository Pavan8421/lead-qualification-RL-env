#!/usr/bin/env bash
# Poll the HF model repo for fresh tensorboard logs (and adapters) every N seconds.
# Usage: ./scripts/sync_tb.sh [interval_seconds] [repo_id]
#
# Logs land under outputs/adapters/tb/ — point tensorboard there.

set -euo pipefail

INTERVAL="${1:-60}"
REPO="${2:-pavanKumar2004/lead-triage-grpo}"
DEST="outputs/adapters"

mkdir -p "$DEST"

echo "[sync_tb] repo=$REPO  interval=${INTERVAL}s  dest=$DEST"
echo "[sync_tb] Ctrl+C to stop."

while true; do
  ts="$(date +%H:%M:%S)"
  if huggingface-cli download "$REPO" \
       --include "tb/**" \
       --local-dir "$DEST" \
       --local-dir-use-symlinks False \
       > /tmp/sync_tb.out 2>&1; then
    n=$(find "$DEST/tb" -name "events.out.tfevents.*" 2>/dev/null | wc -l | tr -d ' ')
    echo "[$ts] synced ($n event files)"
  else
    echo "[$ts] sync failed (repo not ready yet?) — retrying in ${INTERVAL}s"
  fi
  sleep "$INTERVAL"
done
