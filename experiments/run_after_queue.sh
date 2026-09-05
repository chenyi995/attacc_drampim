#!/usr/bin/env bash
# Chain after run_queue.sh: once queue.log says "queue done", run the full
# ladder on the A6 split workload with the prefill-side price log enabled.
# Detached (setsid nohup) so it outlives the session that launched it.
set -u
S=${KVPIM_SCRATCH:-/data2/chenyi9/KV-PIM/scratch_0905}
D=${KVPIM_REPO:-/data2/chenyi9/KV-PIM/attacc_drampim_822}
until grep -q "queue done" "$S/queue.log" 2>/dev/null; do sleep 20; done
cd "$D"
export ATTACC_RAMULATOR_DIR=$S ATTACC_RAMULATOR_LOG=$S/ramulator.out
export KVPIM_CPPCORE=1 PYTHONPATH=$D NUM_HBM=1 NGPU=1 RAMU_WORKERS=9
export RUNGS="A1 A2 A3b A4c A4e A5 A6" KVPIM_PREFILL_SIDE_LOG=$S/side_a6split.jsonl
rm -f "$S/side_a6split.jsonl"; rm -rf "$S/out_a6_split"
echo "$(date +%T) ladder a6_split start" >> "$S/queue.log"
bash experiments/run_dag_ladder.sh "$S/wl/wl_a6_split.json" CACHEBLEND-TINY "$S/out_a6_split" \
    > "$S/ladder_a6_split.log" 2>&1
echo "$(date +%T) ladder a6_split exit $?" >> "$S/queue.log"
