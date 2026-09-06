#!/usr/bin/env bash
# W1 baseline, every combo, every model, one model after another (chenyi9
# 2026-09-06).  Small and mid models: the seven combos in parallel through
# run_sweep.sh (geometry from its table).  LLAMA-65B / GPT-175B: one combo at
# a time (300-460 GB per combo on this host), 16 Ramulator workers.
#   usage: run_w1_all_models.sh <outroot> [models...]
set -u
OUTROOT=${1:?usage: run_w1_all_models.sh <outroot> [models...]}
shift
MODELS=${*:-"LLAMA3-8B LLAMA-7B GPT-13B LLAMA-33B LLAMA-65B GPT-175B"}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$OUTROOT"
for MODEL in $MODELS; do
    echo "$(date +%T) W1 $MODEL start" >> "$OUTROOT/all_models.log"
    case $MODEL in
        LLAMA-65B|GPT-175B)
            for COMBO in A1 A2 A3b A4c A4e A5 A6; do
                RUNGS=$COMBO PARALLEL=1 RAMU_WORKERS=16 SKIP_COLLECT=1 \
                    bash "$SCRIPT_DIR/run_sweep.sh" "$OUTROOT/proto_w1_$MODEL" '^W1_turns' "$MODEL" \
                    > "$OUTROOT/proto_w1_${MODEL}_${COMBO}.out" 2>&1
            done
            python3 "$SCRIPT_DIR/collect_dag_ladder.py" "$OUTROOT/proto_w1_$MODEL/W1_turns" \
                "$SCRIPT_DIR/../workload/probe/sweep/W1_turns.json" "$MODEL" > /dev/null 2>&1 ;;
        *)
            RAMU_WORKERS=8 PARALLEL=1 bash "$SCRIPT_DIR/run_sweep.sh" "$OUTROOT/proto_w1_$MODEL" '^W1_turns' "$MODEL" \
                > "$OUTROOT/proto_w1_$MODEL.out" 2>&1 ;;
    esac
    echo "$(date +%T) W1 $MODEL exit $?" >> "$OUTROOT/all_models.log"
    python3 "$SCRIPT_DIR/extract_protocol.py" "$OUTROOT/proto_w1_$MODEL" --ref A3b > "$OUTROOT/proto_w1_$MODEL/extract.out" 2>&1
done
echo "$(date +%T) W1 all models done" >> "$OUTROOT/all_models.log"
