#!/usr/bin/env bash
# run_tournament.sh
# =================
# 12-model tournament — Normal condition only
# Market coupling: same_sub ×0.70, same_meme ×0.85 (re-enabled)
# 24 specific instances × 1 condition = 1 run
#
# Models (12):
#   gpt5_1  gemini3_1_pro  gemini3_flash  mistral_large  mistral_small_4
#   ministral_14  qwen235b  qwen8b  qwen3_5_27b  qwen3_VL_8B  gemma3_27B  gemma3_12B
#
# Instances (24 by dataset ID):
#   98 96 92 84 83 77 76 71 60 59 58 54 51 44 43 31 30 18 19 10 9 5 57 69
#
# Usage:
#   bash run_tournament.sh
#
# Resume an interrupted run:
#   python -m memeeconomy.run_tournament --resume results/multiagent/run_<timestamp>_tournament_normal

set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

AGENTS=(
    gpt5_1
    gemini3_1_pro
    gemini3_flash
    mistral_large
    mistral_small_4
    ministral_14
    qwen235b
    qwen8b
    qwen3_5_27b
    qwen3_VL_8B
    gemma3_27B
    gemma3_12B
)

ROUNDS=24

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  TOURNAMENT  |  condition=normal  |  agents=${#AGENTS[@]}  |  rounds=$ROUNDS"
echo "  Market coupling: sub ×0.70, meme ×0.85"
echo "  Instances: 24 sampled at random from 98"
echo "════════════════════════════════════════════════════════════════"

python -m memeeconomy.run_tournament \
    --agents "${AGENTS[@]}" \
    --rounds "$ROUNDS" \
    --condition normal \
    --label "tournament_normal"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  TOURNAMENT COMPLETE"
echo "════════════════════════════════════════════════════════════════"
echo "  Next: python evaluation/compute_multiagent_metrics.py results/multiagent/run_*_tournament_normal"
