#!/usr/bin/env bash
# Behavioural removal-rate sweep: full-pressure condition run at 20/40/50/60% removal.
# 30% already exists from the pressure factorial (pf_t50_e1_a1), giving a 5-point curve.
# Same 3 models, same 33 events, same seed. Only the removal probability changes.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY="${PYTHON:-python3}"
export PYTHONHASHSEED=0
LOG=results/removal_behavioral; mkdir -p "$LOG"
read -ra INSTANCES < <($PY -c "import json,random; m=json.load(open('experiments/pressure_factorial_2026/manifest.json')); it=list(m['instances']); random.Random(3530401).shuffle(it); print(' '.join(it))")

run_model(){ MODEL=$1
  for PCT in 20 40 50 60; do
    OUT="results/removal_behavioral/rm${PCT}/${MODEL}"
    if ls "$OUT"/portfolio_run*.json >/dev/null 2>&1; then echo "[rm] $MODEL/$PCT skip"; continue; fi
    echo "[rm] $MODEL / ${PCT}%  ($(date +%H:%M))"
    $PY memeeconomy/run_single_agent.py \
      --models "$MODEL" --rounds 33 --condition pf_t50_e1_a1 --arch bdi \
      --seed 3530401 --config "configs/instance_config_${PCT}pct.json" \
      --run-label "_rm${PCT}" --results-label "removal_behavioral/rm${PCT}" \
      --instances "${INSTANCES[@]}" >> "$LOG/${MODEL}_${PCT}.log" 2>&1 \
      && echo "  ok $MODEL/${PCT}%" || echo "  FAIL $MODEL/${PCT}%"
  done
}
for M in gpt5_1 gemini3_flash qwen32b; do run_model "$M" & done
wait
echo "[rm] ALL DONE ($(date +%H:%M))"
