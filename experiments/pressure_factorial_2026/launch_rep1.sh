#!/usr/bin/env bash
# launch_rep1.sh — single-seed (rep 1) budget launch of the full pressure-factor
# experiment: 9 preregistered conditions + 2 supplementary contrast arms,
# 3 models x 33 instances x 1 rep = 1,089 decisions.
# Core runs first (3 lanes), supplementary arms chained after (2 lanes).
set -uo pipefail
cd "$(dirname "$0")/../.."

PY="${PYTHON:-python3}"
export PYTHONHASHSEED=0

LOGDIR=results/pressure_factorial_2026
mkdir -p "$LOGDIR"

echo "[launch] core factorial rep-1: 27 cells, 891 decisions  ($(date))"
"$PY" experiments/pressure_factorial_2026/run_experiment.py \
    --execute --jobs 3 --repeats 1 \
    > "$LOGDIR/core_rep1.log" 2>&1
CORE_RC=$?
echo "[launch] core finished rc=$CORE_RC  ($(date))"

echo "[launch] supplementary arms rep-1: 6 cells, 198 decisions  ($(date))"
"$PY" experiments/pressure_factorial_2026/run_supplementary.py \
    --jobs 2 --repeats 1 \
    > "$LOGDIR/supp_rep1.log" 2>&1
SUPP_RC=$?
echo "[launch] supplementary finished rc=$SUPP_RC  ($(date))"

echo "[launch] ALL DONE core_rc=$CORE_RC supp_rc=$SUPP_RC  ($(date))"
