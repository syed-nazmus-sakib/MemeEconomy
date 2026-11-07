#!/usr/bin/env bash
# run_all_98.sh
# Full benchmark: all models × all 4 conditions × 98 rounds.
# Runs conditions sequentially per model so each model completes
# its full 4-condition matrix before the next model starts.
#
# Usage:
#   bash run_all_98.sh              # all models
#   bash run_all_98.sh gpt5_1       # single model only
#
# Estimated time: ~6–16 min per model per condition depending on API latency.

set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

ROUNDS=98
START_DESP="results/starting_portfolios/start_desperation.json"
START_EXTR="results/starting_portfolios/start_extremist.json"

# ── Model list ────────────────────────────────────────────────────────────────
ALL_MODELS=(
    "gpt5_1"
    "gemini3_1_pro"
    "gemini2_5_pro"
    "gemini3_flash"
    "gemma4_31B"
    "gemma3_27B"
    "gemma3_4B"
    "qwen235b"
    "qwen32b"
    "qwen8b"
    "qwen3_VL_8B"
    "mistral_large"
    "mistral_small_4"
    "mistral_small_3_2"
    "ministral_14B"
    "ministral_8B"
    "ministral_3B"
)

# If a model name is passed as $1, run only that model.
if [ $# -ge 1 ]; then
    MODELS=("$1")
else
    MODELS=("${ALL_MODELS[@]}")
fi

# ── Generate starting portfolios ──────────────────────────────────────────────
if [ ! -f "${START_DESP}" ] || [ ! -f "${START_EXTR}" ]; then
    echo "Generating starting portfolios for ${ROUNDS} rounds..."
    python scripts/generate_starting_portfolio.py --total-rounds ${ROUNDS}
fi

# ── Helpers ───────────────────────────────────────────────────────────────────
GRAND_PORTFOLIOS=()
FAILED=()

run_condition() {
    local model=$1
    local condition=$2
    local extra_args=${3:-""}

    echo ""
    echo "--------------------------------------------------------"
    echo "  ${model}  |  ${condition}  |  ${ROUNDS} rounds"
    echo "--------------------------------------------------------"

    python -m memeeconomy.run_single_agent \
        --models "${model}" \
        --rounds "${ROUNDS}" \
        --condition "${condition}" \
        ${extra_args} || {
        echo "[WARN] ${model}/${condition} failed. Continuing."
        FAILED+=("${model}/${condition}")
        return
    }

    local portfolio
    portfolio=$(ls -t "results/${model}"/portfolio_*.json 2>/dev/null | head -1)
    if [ -n "${portfolio}" ]; then
        GRAND_PORTFOLIOS+=("${portfolio}")
        python evaluation/evaluate.py "${portfolio}"
    fi
}

# ── Main loop ─────────────────────────────────────────────────────────────────
for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "========================================================"
    echo "  STARTING MODEL: ${MODEL}"
    echo "========================================================"

    mkdir -p "results/${MODEL}"

    run_condition "${MODEL}" "neutral"     ""
    run_condition "${MODEL}" "normal"      ""
    run_condition "${MODEL}" "desperation" "--resume-portfolio ${START_DESP}"
    run_condition "${MODEL}" "extremist"   "--resume-portfolio ${START_EXTR}"

    echo ""
    echo "  [${MODEL}] All conditions done. Comparison:"
    PORTFOLIOS_FOR_MODEL=()
    for cond in neutral normal desperation extremist; do
        p=$(ls -t "results/${MODEL}"/portfolio_*.json 2>/dev/null | head -1)
        [ -n "${p}" ] && PORTFOLIOS_FOR_MODEL+=("${p}")
    done
    [ ${#PORTFOLIOS_FOR_MODEL[@]} -gt 0 ] && \
        python evaluation/evaluate.py "${PORTFOLIOS_FOR_MODEL[@]}" --compare
done

# ── Grand comparison ──────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "  GRAND COMPARISON — all models — all conditions"
echo "========================================================"
if [ ${#GRAND_PORTFOLIOS[@]} -gt 0 ]; then
    python evaluation/evaluate.py "${GRAND_PORTFOLIOS[@]}" --compare
fi

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "[FAILED]: ${FAILED[*]}"
fi

echo ""
echo "Done. Results in results/"
