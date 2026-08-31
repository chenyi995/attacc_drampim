#!/usr/bin/env bash
# Shared helpers: node-local Ramulator working dir + per-model shared cache pool.
REPO=/home/cw636/chenyi/attacc_drampim
# Model table -- identical to experiments/run_sweep_models.sh (ngpu, num_hbm).
declare -A NGPU=( [LLAMA-7B]=1 [LLAMA3-8B]=1 [GPT-13B]=2 [LLAMA-33B]=2 [GPT-175B]=8 [LLAMA-65B]=8 )
declare -A NHBM=( [LLAMA-7B]=1 [LLAMA3-8B]=1 [GPT-13B]=10 [LLAMA-33B]=10 [GPT-175B]=40 [LLAMA-65B]=40 )
RUNGS9="A1 A2 A3 A3a A3b A4 A4b A5 A6"

# make_ramdir <tag> <model> <pool>  -> echoes the dir
# Node-local scratch with room.  /tmp lives on / and is only ~49-62 GB on
# node5/node6 while one trace-heavy rung can need 128 GB -- that is what made
# three rungs die with ENOSPC on 2026-08-31.  /localdata is a real local ext4
# volume with 3-33 TB free on every node but node5, whose /localdata is
# root-owned; there this falls back to /tmp, which is fine because big-model
# tasks are kept off node5 (BIG_EXCLUDE in governor.py).
# Probed on all six nodes before this went in: dir/symlink/exec/writable all ok.
scratch_root() {
  local base=/localdata/kvpim_${USER}
  if mkdir -p "$base" 2>/dev/null && [ -w "$base" ]; then echo "$base"; else echo /tmp; fi
}

make_ramdir() {
  local tag=$1 model=$2 pool=$3
  local root; root=$(scratch_root)
  rm -rf "$root"/kvpim_${USER}_${tag}_* /tmp/kvpim_${USER}_${tag}_* 2>/dev/null
  local rd=$root/kvpim_${USER}_${tag}_$$
  mkdir -p "$rd"
  ln -sf "$REPO/ramulator2/ramulator2" "$rd/ramulator2"
  ln -sf "$REPO/ramulator2/trace_gen"  "$rd/trace_gen"
  cp "$REPO/ramulator.out" "$rd/ramulator.out" 2>/dev/null || :
  # Seed only from THIS model's shards: shapes are disjoint across models
  # (different n_heads / heads_per_hbm), so other models' entries are dead weight.
  cat "$pool/${model}__"*.jsonl > "$rd/signature_cache.jsonl" 2>/dev/null || :
  echo "$rd"
}

# publish_loop <rd> <pool> <model> <tag>   (background; own file per writer)
publish_loop() {
  # DISABLED 2026-08-31: see publish_final.
  return 0
  local rd=$1 pool=$2 model=$3 tag=$4
  while true; do
    sleep 300
    [ -f "$rd/signature_cache.jsonl" ] || continue
    cp "$rd/signature_cache.jsonl" "$pool/.${model}__${tag}.tmp" 2>/dev/null \
      && mv -f "$pool/.${model}__${tag}.tmp" "$pool/${model}__${tag}.jsonl" 2>/dev/null
  done
}

publish_final() {
  # DISABLED 2026-08-31: each publish re-wrote everything the task had SEEDED
  # from, so shards compounded -- 1 MB right after a dedup became 193 GB two
  # hours later, 499 GB across the pool, on a volume at 98%.  A correct version
  # publishes only the lines beyond the seed offset; until that is written and
  # tested, publishing is off.  Memoisation only, so results do not change.
  return 0
  local rd=$1 pool=$2 model=$3 tag=$4
  cp "$rd/signature_cache.jsonl" "$pool/.${model}__${tag}.tmp" 2>/dev/null \
    && mv -f "$pool/.${model}__${tag}.tmp" "$pool/${model}__${tag}.jsonl"
}

export_env() {
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 KVPIM_CPPCORE=1
  # ~16% faster build (verified byte-identical); big-model slots override this to
  # 0 -- uncollected cycles cost them more in parallelism than the 16% is worth.
  export KVPIM_NOGC=${KVPIM_NOGC:-1}
}
