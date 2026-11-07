#!/usr/bin/env bash
# run_react_ablation.sh
# =====================
# ReAct architecture ablation — free-form Thought+Action vs structured BDI
#
# IMPORTANT: All runs use the SAME 18 instances so that condition comparisons
# (normal vs desperation) and architecture comparisons (BDI vs ReAct) are not
# confounded by instance difficulty.
#
# Models     : gpt5_1  gemini3_flash  qwen32b
# Conditions : normal  desperation
# Instances  : 18 fixed (same as multiagent tournament + no-BDI ablation)
# Arch       : react
#
# Total: 3 models × 2 conditions × 18 rounds = 6 runs
#
# After completion compare against BDI runs:
#   python evaluation/evaluate.py results/{gpt5_1,gemini3_flash,qwen32b}/portfolio_run*.json --compare

set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

MODELS=(gpt5_1 gemini3_flash qwen32b)
CONDITIONS=(normal desperation)

# ── FIXED INSTANCE SET ────────────────────────────────────────────────────────
# Same 18 instances used in the multiagent tournament and no-BDI ablation.
INSTANCES=(
    joe_biden_falls_off_bike
    howard_dean_scream
    elon_musk_shirtless_on_a_yacht_in_mykonos
    strait_of_hormuz_bypass_solutions
    bored_ben_affleck_2023_grammys
    pope_francis_puffer_jacket_fake_image
    pope_francis_looking_miserable_next_to_trump
    joe_biden_obama_bromance
    nigel_farage_milkshakeing_aftermath
    theresa_may_dancing
    king_charles_sausage_fingers
    trump_with_the_glowing_orb_in_saudi_arabia
    suez_canal_evergreen_stuck
    will_smith_slaps_chris_rock
    turkish_shooter_aura
    prince_andrew_arrested
    palestine_vs_israel
    india_ai_show_scam_galgotiya_uni
)
# ─────────────────────────────────────────────────────────────────────────────

ROUNDS=${#INSTANCES[@]}   # 18

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  ReAct Ablation"
echo "  Models     : ${#MODELS[@]}"
echo "  Conditions : ${CONDITIONS[*]}"
echo "  Instances  : $ROUNDS fixed (same for every run)"
echo "════════════════════════════════════════════════════════════"

for MODEL in "${MODELS[@]}"; do
    for COND in "${CONDITIONS[@]}"; do
        echo ""
        echo "  ── model=$MODEL  condition=$COND  rounds=$ROUNDS ──"
        python -m memeeconomy.run_single_agent \
            --models "$MODEL" \
            --rounds "$ROUNDS" \
            --condition "$COND" \
            --arch react \
            --instances "${INSTANCES[@]}"
        echo "  ✅  $MODEL / $COND done."
    done
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  ALL ReAct RUNS COMPLETE"
echo "════════════════════════════════════════════════════════════"
echo "  Next: python evaluation/evaluate.py results/{gpt5_1,gemini3_flash,qwen32b}/portfolio_run*.json --compare"
