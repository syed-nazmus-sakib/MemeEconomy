#!/usr/bin/env bash
# Behavioural check at 50% Tier-5 removal. Full-pressure condition, 19 events.
# Compared against the SAME 19 events from the existing 30% factorial run (paired).
set -uo pipefail
cd "$(dirname "$0")/../.."
PY="${PYTHON:-python3}"
export PYTHONHASHSEED=0
LOG=results/removal_behavioral; mkdir -p "$LOG"
read -ra INST < <($PY -c "import json,random; m=json.load(open('experiments/pressure_factorial_2026/manifest.json')); it=list(m['instances']); random.Random(3530401).shuffle(it); print(' '.join(it[:19]))")
echo "events: ${#INST[@]}"
run(){ MODEL=$1
  if ls results/removal_behavioral/rm50/$MODEL/portfolio_run*.json >/dev/null 2>&1; then echo "[rm50] $MODEL skip"; return; fi
  echo "[rm50] $MODEL ($(date +%H:%M))"
  $PY memeeconomy/run_single_agent.py \
    --models "$MODEL" --rounds 19 --condition pf_t50_e1_a1 --arch bdi \
    --seed 3530401 --config configs/instance_config_50pct.json \
    --run-label "_rm50" --results-label "removal_behavioral/rm50" \
    --instances "${INST[@]}" >> "$LOG/${MODEL}_50.log" 2>&1 \
    && echo "  ok $MODEL" || echo "  FAIL $MODEL"
}
for M in gpt5_1 gemini3_flash qwen32b; do run "$M" & done
wait
echo "[rm50] ALL DONE ($(date +%H:%M))"
