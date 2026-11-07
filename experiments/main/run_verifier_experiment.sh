#!/usr/bin/env bash
# run_verifier_experiment.sh
# ==========================
# Real-time Verifier Experiment — GPT-5.1 as verifier
#
# Run A: Gemini 3 Flash (subject)  ←  GPT-5.1 (verifier)
# Run B: Qwen3-32B    (subject)    ←  GPT-5.1 (verifier)
#
# Both use the SAME 12 fixed instances.
# Condition: normal

set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

INSTANCES=(
    joe_biden_falls_off_bike
    howard_dean_scream
    elon_musk_shirtless_on_a_yacht_in_mykonos
    strait_of_hormuz_bypass_solutions
    bored_ben_affleck_2023_grammys
    pope_francis_puffer_jacket_fake_image
    joe_biden_obama_bromance
    nigel_farage_milkshakeing_aftermath
    theresa_may_dancing
    will_smith_slaps_chris_rock
    prince_andrew_arrested
    palestine_vs_israel
)

ROUNDS=${#INSTANCES[@]}   # 12
COND=normal
VERIFIER=gpt5_1

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Verifier Experiment  |  Verifier: $VERIFIER"
echo "  Condition: $COND  |  Rounds: $ROUNDS"
echo "════════════════════════════════════════════════════════════"

echo ""
echo "  RUN A — Subject: gemini3_flash  |  Verifier: $VERIFIER"
echo "  ─────────────────────────────────────────────────────────"
python -m memeeconomy.run_single_agent \
    --models gemini3_flash \
    --rounds "$ROUNDS" \
    --condition "$COND" \
    --verifier "$VERIFIER" \
    --instances "${INSTANCES[@]}"
echo "  ✅  Run A complete."

echo ""
echo "  RUN B — Subject: qwen32b  |  Verifier: $VERIFIER"
echo "  ─────────────────────────────────────────────────────────"
python -m memeeconomy.run_single_agent \
    --models qwen32b \
    --rounds "$ROUNDS" \
    --condition "$COND" \
    --verifier "$VERIFIER" \
    --instances "${INSTANCES[@]}"
echo "  ✅  Run B complete."

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  VERIFIER EXPERIMENT COMPLETE"
echo "════════════════════════════════════════════════════════════"
