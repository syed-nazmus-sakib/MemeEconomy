#!/usr/bin/env bash
# run_ablation2_nobdi.sh
# ======================
# Ablation 2: BDI Schema Removal
#
# Defends Finding 2 against "prompt artifact" reviewer attack.
# If T5 rates remain elevated under Desperation without the BDI schema,
# the suppression finding is behavioural, not a prompt structure artifact.
#
# Design:
#   - Architecture  : nobdi (3-field output: meme + subreddit + risk_acknowledged)
#   - Conditions    : normal  and  desperation
#   - Models        : all 10 Table-1 models
#   - Instances     : 13 fixed (random seed=42 from 98 pool) — same for ALL runs
#   - Rounds        : 13 (one per instance)
#
# Fixed instance set (seed=42):
#   new_leo_meme_tfw_you_didn_t_agree_to_this
#   fyre_festival_disaster
#   bernie_sanders_mitten_hand
#   mark_zuckerberg_congressional_hearings
#   south_korea_pres_jailed
#   punch_the_monkey
#   nancy_pelosi_clap
#   india_ai_show_scam_galgotiya_uni
#   ezra_miller_s_crime_spree
#   rudy_giulianis_hair_dye_streaks
#   hillary_clintons_texts_from_hillary
#   elon_musk_naming_his_kid
#   pope_francis_looking_miserable_next_to_trump
#
# Results land in: results/<model>/portfolio_run*_nobdi_*.json
# Compare against BDI runs by filtering bdi_raw._arch == "nobdi"

set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

INSTANCES=(
    new_leo_meme_tfw_you_didn_t_agree_to_this
    fyre_festival_disaster
    bernie_sanders_mitten_hand
    mark_zuckerberg_congressional_hearings
    south_korea_pres_jailed
    punch_the_monkey
    nancy_pelosi_clap
    india_ai_show_scam_galgotiya_uni
    ezra_miller_s_crime_spree
    rudy_giulianis_hair_dye_streaks
    hillary_clintons_texts_from_hillary
    elon_musk_naming_his_kid
    pope_francis_looking_miserable_next_to_trump
)

MODELS=(
    gpt5_1
    gemini3_1_pro
    gemini3_flash
    mistral_large
    ministral_14
    qwen235b
    qwen32b
    qwen8b
    gemma3_12B
    gemma3_27B
)

ROUNDS=${#INSTANCES[@]}    # 13
DESP_PORTFOLIO="results/starting_portfolios/start_desperation.json"
FAILED=()

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Ablation 2: BDI Schema Removal"
echo "  Architecture : nobdi"
echo "  Models       : ${#MODELS[@]}"
echo "  Instances    : $ROUNDS fixed (seed=42)"
echo "  Conditions   : normal + desperation"
echo "════════════════════════════════════════════════════════════════"

# ── Helper ────────────────────────────────────────────────────────────────────
run_one() {
    local MODEL="$1"
    local COND="$2"
    local EXTRA="${3:-}"

    echo ""
    echo "  ── $MODEL | $COND | nobdi ──"

    CMD=(python -m memeeconomy.run_single_agent
        --models    "$MODEL"
        --rounds    "$ROUNDS"
        --condition "$COND"
        --arch      nobdi
        --run-label _abl2
        --instances "${INSTANCES[@]}")

    # Desperation: load the shared starting portfolio so karma pressure is real
    if [[ "$COND" == "desperation" ]]; then
        CMD+=(--resume-portfolio "$DESP_PORTFOLIO")
    fi

    if "${CMD[@]}"; then
        echo "  ✅  $MODEL / $COND done."
    else
        echo "  ❌  $MODEL / $COND FAILED"
        FAILED+=("$MODEL/$COND")
    fi
}

# ── Normal condition — all 10 models ─────────────────────────────────────────
echo ""
echo "  ┌─────────────────────────────────────────────────────────┐"
echo "  │  PASS 1 — Normal condition                              │"
echo "  └─────────────────────────────────────────────────────────┘"

for MODEL in "${MODELS[@]}"; do
    run_one "$MODEL" normal
done

# ── Desperation condition — all 10 models ────────────────────────────────────
echo ""
echo "  ┌─────────────────────────────────────────────────────────┐"
echo "  │  PASS 2 — Desperation condition                         │"
echo "  └─────────────────────────────────────────────────────────┘"

for MODEL in "${MODELS[@]}"; do
    run_one "$MODEL" desperation
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ABLATION 2 COMPLETE"
echo "  Total runs: $((${#MODELS[@]} * 2))  (10 models × 2 conditions)"
echo "════════════════════════════════════════════════════════════════"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "  ⚠️  Failed runs: ${FAILED[*]}"
else
    echo "  ✅  All runs succeeded."
fi

echo ""
echo "  Next: python evaluation/evaluate.py results/<model>/portfolio_run*.json --compare"
echo ""
echo "  To find results:"
echo "    ls results/gpt5_1/portfolio_run*nobdi* 2>/dev/null || \\"
echo "    python3 -c \""
echo "      import json, glob"
echo "      for f in sorted(glob.glob('results/*/portfolio_run*.json'))[-20:]:"
echo "          d = json.load(open(f))"
echo "          if any('nobdi' in str(r.get('bdi_raw','')) or"
echo "                  r.get('arch','') == 'nobdi'"
echo "                  for r in d.get('round_history',[])):"
echo "              print(f)\""
