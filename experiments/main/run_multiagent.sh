#!/usr/bin/env bash
# run_multiagent.sh
# =================
# Multi-agent competitive benchmark experiments for the NeurIPS paper.
#
# Experiment matrix (6 runs):
#   1. HOMO-NORMAL      : 2× gpt5_1,       normal      (competition baseline)
#   2. HOMO-DESP        : 2× gpt5_1,       desperation (competition + pressure)
#   3. HETERO-NORMAL    : gpt5_1 vs gemini25_pro, normal
#   4. HETERO-DESP      : gpt5_1 vs gemini25_pro, desperation
#   5. 4WAY-NORMAL      : 4 models,        normal      (max competition)
#   6. 4WAY-DESP        : 4 models,        desperation (max competition + pressure)
#
# Usage:
#   bash run_multiagent.sh              # all 6 experiments
#   bash run_multiagent.sh homo         # only homogeneous runs (1+2)
#   bash run_multiagent.sh hetero       # only heterogeneous runs (3+4)
#   bash run_multiagent.sh 4way         # only 4-way runs (5+6)
#   bash run_multiagent.sh homo_normal  # single specific run
#
# Results go to: results/multiagent/run_<timestamp>_<models>/

set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

ROUNDS=46   # matches single-agent benchmark

FAILED=()

run_exp() {
    local label="$1"
    local condition="$2"
    shift 2
    local agents=("$@")

    echo ""
    echo "============================================================"
    echo "  EXPERIMENT: ${label}"
    echo "  Agents    : ${agents[*]}"
    echo "  Condition : ${condition}"
    echo "  Rounds    : ${ROUNDS}"
    echo "============================================================"

    python -m memeeconomy.run_tournament \
        --agents "${agents[@]}" \
        --rounds "${ROUNDS}" \
        --condition "${condition}" \
        || {
            echo "[WARN] ${label} failed. Continuing."
            FAILED+=("${label}")
        }
}

# ── Experiment selector ───────────────────────────────────────────────────────
MODE="${1:-all}"

run_homo() {
    run_exp "HOMO-NORMAL"      "normal"      gpt5_1 gpt5_1
    run_exp "HOMO-DESP"        "desperation" gpt5_1 gpt5_1
}

run_hetero() {
    run_exp "HETERO-NORMAL"    "normal"      gpt5_1 gemini3_1_pro
    run_exp "HETERO-DESP"      "desperation" gpt5_1 gemini3_1_pro
}

run_4way() {
    run_exp "4WAY-NORMAL"      "normal"      gpt5_1 gemini3_1_pro qwen235b mistral_large
    run_exp "4WAY-DESP"        "desperation" gpt5_1 gemini3_1_pro qwen235b mistral_large
}

case "${MODE}" in
    homo)        run_homo ;;
    hetero)      run_hetero ;;
    4way)        run_4way ;;
    homo_normal) run_exp "HOMO-NORMAL"  "normal"      gpt5_1 gpt5_1 ;;
    homo_desp)   run_exp "HOMO-DESP"   "desperation" gpt5_1 gpt5_1 ;;
    hetero_normal) run_exp "HETERO-NORMAL" "normal"   gpt5_1 gemini3_1_pro ;;
    hetero_desp)   run_exp "HETERO-DESP"  "desperation" gpt5_1 gemini3_1_pro ;;
    4way_normal) run_exp "4WAY-NORMAL" "normal"      gpt5_1 gemini3_1_pro qwen235b mistral_large ;;
    4way_desp)   run_exp "4WAY-DESP"   "desperation" gpt5_1 gemini3_1_pro qwen235b mistral_large ;;
    all)
        run_homo
        run_hetero
        run_4way
        ;;
    *)
        echo "Unknown mode: ${MODE}"
        echo "Usage: bash run_multiagent.sh [all|homo|hetero|4way|homo_normal|homo_desp|hetero_normal|hetero_desp|4way_normal|4way_desp]"
        exit 1
        ;;
esac

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  MULTI-AGENT RUNS COMPLETE"
echo "  Results in: results/multiagent/"
echo "============================================================"

# List all multi-agent summary files
if ls results/multiagent/*/summary.json &>/dev/null; then
    echo ""
    echo "  Summaries:"
    for f in results/multiagent/*/summary.json; do
        run=$(dirname "$f" | xargs basename)
        karma_vals=$(python3 -c "
import json, sys
d = json.load(open('${f}'))
lb = d.get('final_leaderboard', [])
parts = [f\"  #{e['rank']} {e['model_key']}: {e['karma']:,} karma, T5={e['tier5_rate']}%\" for e in lb]
print('\n'.join(parts))
" 2>/dev/null || echo "    (parse error)")
        echo "  ── ${run}"
        echo "${karma_vals}"
    done
fi

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "[FAILED]: ${FAILED[*]}"
fi

echo ""
echo "Done."
