#!/usr/bin/env bash
set -euo pipefail
cd /home/thanhtung/projects/sam-bioclip-agriculture
PY=/home/thanhtung/projects/my-ai-project/.venv/bin/python
CODE=experiments/revision_v3
RESULTS=experiments/results/revision_v3_20261104
mkdir -p "$RESULTS"
exec 9>"$RESULTS/queue.lock"
flock -n 9 || { echo 'Queue already running'; exit 1; }
trap 'echo "FAILED $(date -Iseconds) line $LINENO" > "$RESULTS/queue_status.txt"' ERR
for phase in lr fewshot; do
  for action in plan train select; do
    echo "$phase $action $(date -Iseconds)" | tee "$RESULTS/queue_status.txt"
    "$PY" "$CODE/revision_v3.py" "$action" --phase "$phase" --results "$RESULTS"
  done
done
for phase in lr fewshot; do
  echo "$phase evaluate $(date -Iseconds)" | tee "$RESULTS/queue_status.txt"
  "$PY" "$CODE/revision_v3.py" evaluate --phase "$phase" --results "$RESULTS"
done
echo "COMPLETE lr and fewshot $(date -Iseconds)" | tee "$RESULTS/queue_status.txt"
