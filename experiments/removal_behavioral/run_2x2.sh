#!/usr/bin/env bash
# Completes the 2x2: {neutral, full pressure} x {30%, 50% removal}.
#   30% cells already exist in results/pressure_factorial_2026 (neutral, pf_t50_e1_a1).
#   Part A: extend full-pressure @50% from 19 to 33 events by RESUMING the existing
#           19-round portfolio, so karma continues rather than restarting at 15k.
#   Part B: neutral @50%, fresh 33-event run.
# Runs 2 lanes only, since another experiment is using the browser concurrently.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY="${PYTHON:-python3}"
export PYTHONHASHSEED=0
LOG=results/removal_behavioral; mkdir -p "$LOG"
read -ra INST < <($PY -c "import json,random; m=json.load(open('experiments/pressure_factorial_2026/manifest.json')); it=list(m['instances']); random.Random(3530401).shuffle(it); print(' '.join(it))")

partA(){  # extend full pressure @50% to 33 events, resuming the 19-round portfolio
  for M in gpt5_1 gemini3_flash qwen32b; do
    P=$(ls -t results/removal_behavioral/rm50/$M/portfolio_run*.json 2>/dev/null | head -1)
    R=$($PY -c "import json;print(len(json.load(open('$P'))['round_history']))" 2>/dev/null || echo 0)
    if [ "$R" -ge 33 ]; then echo "[A] $M already 33 rounds, skip"; continue; fi
    echo "[A] $M extending from $R rounds ($(date +%H:%M))"
    $PY memeeconomy/run_single_agent.py --models "$M" --rounds 33 --condition pf_t50_e1_a1 \
      --arch bdi --seed 3530401 --config configs/instance_config_50pct.json \
      --resume-portfolio "$P" --run-label "_rm50ext" --results-label "removal_behavioral/rm50" \
      --instances "${INST[@]}" >> "$LOG/${M}_50ext.log" 2>&1 \
      && echo "[A] ok $M" || echo "[A] FAIL $M"
  done
}
partB(){  # neutral @50%, fresh
  for M in gpt5_1 gemini3_flash qwen32b; do
    if ls results/removal_behavioral/neutral50/$M/portfolio_run*.json >/dev/null 2>&1; then
      echo "[B] $M skip"; continue; fi
    echo "[B] $M neutral@50 ($(date +%H:%M))"
    $PY memeeconomy/run_single_agent.py --models "$M" --rounds 33 --condition neutral \
      --arch bdi --seed 3530401 --config configs/instance_config_50pct.json \
      --run-label "_neu50" --results-label "removal_behavioral/neutral50" \
      --instances "${INST[@]}" >> "$LOG/${M}_neu50.log" 2>&1 \
      && echo "[B] ok $M" || echo "[B] FAIL $M"
  done
}
partA & partB & wait
echo "[2x2] ALL DONE ($(date +%H:%M))"
