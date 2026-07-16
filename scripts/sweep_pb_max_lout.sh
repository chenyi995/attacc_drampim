#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL="${MODEL:-GPT-175B}"
GPU="${GPU:-A100a}"
NGPU="${NGPU:-8}"
GMEMCAP="${GMEMCAP:-80}"
PIM="${PIM:-bank}"
LIN="${LIN:-2048}"
BATCH="${BATCH:-1}"
NUM_AGENT="${NUM_AGENT:-256}"
TOKEN_BLOCK="${TOKEN_BLOCK:-32}"
SIM_CORES="${SIM_CORES:-8}"
MAX_LOUT_SEARCH_LIMIT="${MAX_LOUT_SEARCH_LIMIT:-1048576}"
OUT_ROOT="${OUT_ROOT:-${REPO_DIR}/outputs/pb_sweep_max_lout}"
RUN_NAME="${RUN_NAME:-model_${MODEL}_lin_${LIN}_batch_${BATCH}_agent_${NUM_AGENT}_blk_${TOKEN_BLOCK}_cores_${SIM_CORES}}"
RUN_DIR="${OUT_ROOT}/${RUN_NAME}"

mkdir -p "${RUN_DIR}"

for pb_int in 5 10 15 20 25 30 35 40 45 50 55 60; do
  pb="0.$(printf '%02d' "${pb_int}")"
  pb_tag="pb_${pb/./p}"
  pb_dir="${RUN_DIR}/${pb_tag}"
  mkdir -p "${pb_dir}"

  output_file="${pb_dir}/max_lout_${pb_tag}.csv"
  log_file="${pb_dir}/run_${pb_tag}.log"

  echo "[Pb=${pb}] writing ${output_file}"
  (
    cd "${REPO_DIR}"
    python3 main.py \
      --system dgx-attacc \
      --gpu "${GPU}" \
      --ngpu "${NGPU}" \
      --gmemcap "${GMEMCAP}" \
      --pim "${PIM}" \
      --model "${MODEL}" \
      --lin "${LIN}" \
      --batch "${BATCH}" \
      --num-agent "${NUM_AGENT}" \
      --pb "${pb}" \
      --token-block "${TOKEN_BLOCK}" \
      --sim-cores "${SIM_CORES}" \
      --find-max-lout \
      --max-lout-search-limit "${MAX_LOUT_SEARCH_LIMIT}" \
      --output-file "${output_file}"
  ) > "${log_file}" 2>&1

done

echo "Sweep complete: ${RUN_DIR}"
