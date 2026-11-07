#!/usr/bin/env bash
# run_ablation4_robustness.sh
# ============================
# Ablation 4: Sample Size Robustness
#
# Runs each model+condition 3 independent times on the same 7 fixed instances.
# Measures run-to-run variance in T5%, GF%, and PIM to provide error bars
# for Table 1 — directly addresses the "no confidence intervals" reviewer concern.
#
# Design:
#   Models     : gpt5_1  gemini3_flash  qwen235b  mistral_large  qwen32b  gemma3_27B
#   Conditions : normal  desperation
#   Instances  : 7 fixed (seed=99)
#   Rounds     : 7 per run
#   Repeats    : 3 independent runs per model+condition
#   Total runs : 6 × 2 × 3 = 36
#   Total rounds: 36 × 7 = 252
#
# Fixed instance set (seed=99):
#   fell_out_of_a_coconut_tree_kamala_harris
#   dark_brandon
#   mark_zuckerberg_s_metaverse_avatar
#   pope_francis_puffer_jacket_fake_image
#   kim_jong_un_feeling_left_out
#   palestine_vs_israel
#   punch_the_monkey

set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

INSTANCES=(
    fell_out_of_a_coconut_tree_kamala_harris
    dark_brandon
    mark_zuckerberg_s_metaverse_avatar
    pope_francis_puffer_jacket_fake_image
    kim_jong_un_feeling_left_out
    palestine_vs_israel
    punch_the_monkey
)

MODELS=(
    # gpt5_1     ← already done via run_ablation4_gpt_test.sh — uncomment to re-run
    gemini3_flash
    qwen235b
    mistral_large
    qwen32b
    gemma3_27B
)

ROUNDS=${#INSTANCES[@]}    # 7
DESP_PORTFOLIO="results/starting_portfolios/start_desperation.json"
FAILED=()

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Ablation 4: Sample Size Robustness"
echo "  Models     : ${#MODELS[@]}"
echo "  Conditions : normal + desperation"
echo "  Instances  : $ROUNDS fixed (seed=99)"
echo "  Repeats    : 3 per model+condition"
echo "  Total runs : $((${#MODELS[@]} * 2 * 3))  ($((${#MODELS[@]} * 2 * 3 * ROUNDS)) rounds)"
echo "════════════════════════════════════════════════════════════════"

run_one() {
    local MODEL="$1"
    local COND="$2"
    local REP="$3"

    echo "  ── $MODEL | $COND | rep $REP ──"

    CMD=(python -m memeeconomy.run_single_agent
        --models         "$MODEL"
        --rounds         "$ROUNDS"
        --condition      "$COND"
        --arch           bdi
        --run-label      _abl4
        --results-label  abl4
        --instances      "${INSTANCES[@]}")

    # NOTE: no --resume-portfolio here — desperation framing comes from the prompt
    # (50k target vs 15k balance = sufficient pressure, and all 7 instances run fresh)

    if "${CMD[@]}"; then
        echo "  ✅  $MODEL / $COND / rep $REP done."
    else
        echo "  ❌  $MODEL / $COND / rep $REP FAILED"
        FAILED+=("$MODEL/$COND/rep$REP")
    fi
}

# ── 3 passes — each pass = one complete run of all 12 model×condition combos ──

for REP in 1 2 3; do
    echo ""
    echo "  ┌─────────────────────────────────────────────────────────┐"
    printf  "  │  REPETITION %d of 3                                     │\n" "$REP"
    echo "  └─────────────────────────────────────────────────────────┘"

    for MODEL in "${MODELS[@]}"; do
        run_one "$MODEL" normal      "$REP"
        run_one "$MODEL" desperation "$REP"
    done
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ABLATION 4 COMPLETE"
echo "  Total: $((${#MODELS[@]} * 2 * 3)) runs  ($((${#MODELS[@]} * 2 * 3 * ROUNDS)) rounds)"
echo "════════════════════════════════════════════════════════════════"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "  ⚠️  Failed: ${FAILED[*]}"
else
    echo "  ✅  All runs succeeded."
fi

echo ""
echo "  Next: python evaluation/evaluate.py results/abl4/*/portfolio_*.json --compare"
