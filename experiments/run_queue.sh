#!/usr/bin/env bash
# Re-run queue after the num_attacc fix (fc0216d).  Core budget <= 64:
#   a full ladder = 6 PIM rungs x W Ramulator workers + 7 construction procs
#   the A6 probe   = 2 x W + 3
# Pair A: R8C2N8_L128 (full) + A6 probe          -> 6W+7 + 2W+3 = 42 at W=4
# Pair B: R16C2N16_L128 (full) + R8C2N8_L1024   -> 2 x (6W+7)   = 62 at W=4
set -u
S=${KVPIM_SCRATCH:-/data2/chenyi9/KV-PIM/scratch_0905}
D=${KVPIM_REPO:-/data2/chenyi9/KV-PIM/attacc_drampim_822}
export ATTACC_RAMULATOR_DIR=$S ATTACC_RAMULATOR_LOG=$S/ramulator.out
export KVPIM_CPPCORE=1 PYTHONPATH=$D NUM_HBM=1 NGPU=1 RAMU_WORKERS=4
cd "$D"
ladder() {   # <tag> <workload> <rungs>
    local tag=$1 wl=$2 rungs=$3
    rm -rf "$S/out_$tag"
    RUNGS="$rungs" bash experiments/run_dag_ladder.sh "$S/wl/$wl" CACHEBLEND-TINY "$S/out_$tag" \
        > "$S/ladder_$tag.log" 2>&1
    echo "$(date +%T) ladder $tag exit $?" >> "$S/queue.log"
}
echo "$(date +%T) queue start (pair A)" >> "$S/queue.log"
ladder R8C2N8_L128     wl_mr_R8C2N8_L128.json        "A1 A2 A3b A4c A4e A5 A6" &
ladder a6_own16_256    wl_a6_R8C2N8_own16_256.json   "A4e A5 A6" &
wait
echo "$(date +%T) pair A done; pair B" >> "$S/queue.log"
ladder R16C2N16_L128   wl_mr_R16C2N16_L128.json      "A1 A2 A3b A4c A4e A5 A6" &
ladder R8C2N8_L1024    wl_mr_R8C2N8_L1024.json       "A1 A2 A3b A4c A4e A5 A6" &
wait
echo "$(date +%T) queue done" >> "$S/queue.log"
