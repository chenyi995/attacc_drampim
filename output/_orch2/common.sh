#!/usr/bin/env bash
# Shared helpers: node-local Ramulator working dir + per-model shared cache pool.
REPO=/home/cw636/chenyi/attacc_drampim
# Model table -- identical to experiments/run_sweep_models.sh (ngpu, num_hbm).
declare -A NGPU=( [LLAMA-7B]=1 [LLAMA3-8B]=1 [GPT-13B]=2 [LLAMA-33B]=2 [GPT-175B]=8 [LLAMA-65B]=8 )
declare -A NHBM=( [LLAMA-7B]=1 [LLAMA3-8B]=1 [GPT-13B]=10 [LLAMA-33B]=10 [GPT-175B]=40 [LLAMA-65B]=40 )
RUNGS9="A1 A2 A3 A3a A3b A4 A4b A5 A6"

# make_ramdir <tag> <model> <pool>  -> echoes the dir
make_ramdir() {
  local tag=$1 model=$2 pool=$3
  rm -rf /tmp/kvpim_${USER}_${tag}_* 2>/dev/null
  local rd=/tmp/kvpim_${USER}_${tag}_$$
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
  local rd=$1 pool=$2 model=$3 tag=$4
  while true; do
    sleep 300
    [ -f "$rd/signature_cache.jsonl" ] || continue
    cp "$rd/signature_cache.jsonl" "$pool/.${model}__${tag}.tmp" 2>/dev/null \
      && mv -f "$pool/.${model}__${tag}.tmp" "$pool/${model}__${tag}.jsonl" 2>/dev/null
  done
}

publish_final() {
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
