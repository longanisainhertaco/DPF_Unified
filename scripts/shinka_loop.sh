#!/bin/bash
# Unattended ShinkaEvolve loop with auto-restart on failure.
# Runs continuously until STOP file is created or max iterations reached.
#
# Usage: tmux new-session -d -s shinka './scripts/shinka_loop.sh'
# Stop:  touch ~/dpf-unified/STOP_SHINKA
# Check: tmux attach -t shinka

set -e
cd /Users/anthonyzamora/dpf-unified
source ~/.zshenv 2>/dev/null

if [ -z "${OPENAI_API_KEY:-}" ]; then
    export SHINKA_EMBEDDINGS_DISABLED=1
    echo "OPENAI_API_KEY unset; remote embedding calls disabled for this loop."
fi

MAX_ROUNDS=100
GENS_PER_ROUND=50
ROUND=1

# Caffeinate: prevent sleep during runs
caffeinate -dims &
CAFF_PID=$!
trap "kill $CAFF_PID 2>/dev/null" EXIT

echo "=== ShinkaEvolve Unattended Loop ==="
echo "Rounds: $MAX_ROUNDS x $GENS_PER_ROUND generations"
echo "Stop: touch ~/dpf-unified/STOP_SHINKA"
echo "Monitor: tail -f output/shinka_latest/evolution_run.log"
echo ""

while [ $ROUND -le $MAX_ROUNDS ]; do
    if [ -f STOP_SHINKA ]; then
        echo "STOP_SHINKA file found. Exiting gracefully."
        rm -f STOP_SHINKA
        break
    fi

    RESULT_DIR="output/shinka_round_${ROUND}"
    echo "[$(date)] Starting round $ROUND -> $RESULT_DIR"

    # Symlink for easy monitoring
    ln -sfn "$RESULT_DIR" output/shinka_latest

    # Use Ollama Qwen-Coder as primary (free, local), Claude as fallback.
    # Ollama serves OpenAI-compatible API at port 11434.
    shinka_run \
        --task-dir shinka_dpf \
        --results_dir "$RESULT_DIR" \
        --num_generations $GENS_PER_ROUND \
        --max-evaluation-jobs 2 \
        --set 'evo.llm_models=["claude-4-sonnet-20250514"]' \
        --set 'evo.code_embed_sim_threshold=1.0' \
        --set "evo.task_sys_msg=You are optimizing an MHD z-pinch solver for a PF-1000 dense plasma focus. Target: I_peak=1.82 MA at t_peak=6.3 us. Current: I_peak~2.0 MA (over), t_peak~8-10 us (late). EVOLVE-BLOCK has: GRID_NR (32-128), GRID_NZ (32-128), FC (0.5-0.9, current fraction), FM (0.05-0.25, mass fraction), PRESSURE_TORR (2-6), V0_KV (20-35). Published values: FC=0.7, FM=0.13, P=3.5 Torr, V0=27 kV. Higher FC = more current in sheath = more compression = lower I_peak. Higher FM = more mass swept = slower sheath = later t_peak. Do NOT modify anything outside the EVOLVE-BLOCK markers." \
        --verbose 2>&1 | tee "$RESULT_DIR.log"

    EXIT_CODE=$?
    echo "[$(date)] Round $ROUND completed (exit code: $EXIT_CODE)"

    # Extract best score
    BEST=$(sqlite3 "$RESULT_DIR/programs.sqlite" "SELECT MAX(score) FROM programs WHERE is_correct=1" 2>/dev/null || echo "0")
    echo "[$(date)] Best fitness this round: $BEST"

    # If a correct program was found with fitness > 0.9, we're done
    if [ "$(echo "$BEST > 0.9" | bc -l 2>/dev/null)" = "1" ]; then
        echo "SUCCESS: Fitness > 0.9 achieved in round $ROUND!"
        break
    fi

    ROUND=$((ROUND + 1))
    sleep 10  # brief pause between rounds
done

echo "[$(date)] ShinkaEvolve loop finished after $((ROUND - 1)) rounds."
kill $CAFF_PID 2>/dev/null
