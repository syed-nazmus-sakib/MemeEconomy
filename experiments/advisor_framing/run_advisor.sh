#!/usr/bin/env bash
# ADVISOR framing: model advises a client under full pressure and holds no stake.
# Paired against pf_t50_e1_a1 (identical target, elimination rule, aggressive framing,
# same 33 events, same seeds). Only WHO owns the decision differs.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PYTHON:-python}
export PYTHONHASHSEED=0
LOG=results/advisor_framing; mkdir -p "$LOG"
read -ra INSTANCES < <($PY -c "import json,random; m=json.load(open('experiments/pressure_factorial_2026/manifest.json')); it=list(m['instances']); random.Random(3530401).shuffle(it); print(' '.join(it))")

run(){ MODEL=$1
  if ls results/advisor_framing/advisor/$MODEL/portfolio_run*.json >/dev/null 2>&1; then
    echo "[adv] $MODEL already done, skip"; return; fi
  echo "[adv] $MODEL  ($(date +%H:%M))"
  $PY -m memeeconomy.run_single_agent \
    --models "$MODEL" --rounds 33 --condition advisor --arch bdi \
    --seed 3530401 --run-label "_adv" --results-label "advisor_framing/advisor" \
    --instances "${INSTANCES[@]}" >> "$LOG/${MODEL}.log" 2>&1 \
    && echo "  ok $MODEL" || echo "  FAIL $MODEL"
}
for M in gpt5_1 gemini3_flash qwen32b; do run "$M" & done
wait
echo "[adv] ALL DONE ($(date +%H:%M))"
