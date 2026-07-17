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
LOUT="${LOUT:-128}"
BATCH="${BATCH:-1}"
TOKEN_BLOCK="${TOKEN_BLOCK:-32}"
SIM_CORES="${SIM_CORES:-8}"
OUT_ROOT="${OUT_ROOT:-${REPO_DIR}/outputs/0718/lout128}"
RUN_NAME="${RUN_NAME:-model_${MODEL}_lin_${LIN}_lout_${LOUT}_batch_${BATCH}_blk_${TOKEN_BLOCK}_cores_${SIM_CORES}}"
RUN_DIR="${OUT_ROOT}/${RUN_NAME}"

mkdir -p "${RUN_DIR}"

for agent in 64 256; do
  agent_dir="${RUN_DIR}/agent_${agent}"
  mkdir -p "${agent_dir}"

  for pb_int in 5 10 15 20 25 30 35 40 45 50 55 60; do
    pb="0.$(printf '%02d' "${pb_int}")"
    pb_tag="pb_${pb/./p}"
    case_dir="${agent_dir}/${pb_tag}"
    mkdir -p "${case_dir}"

    output_file="${case_dir}/output_agent_${agent}_${pb_tag}_lout_${LOUT}.csv"
    log_file="${case_dir}/run_agent_${agent}_${pb_tag}_lout_${LOUT}.log"

    echo "[agent=${agent} Pb=${pb} lout=${LOUT}] writing ${output_file}"
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
        --lout "${LOUT}" \
        --batch "${BATCH}" \
        --num-agent "${agent}" \
        --pb "${pb}" \
        --token-block "${TOKEN_BLOCK}" \
        --sim-cores "${SIM_CORES}" \
        --output-file "${output_file}"
    ) > "${log_file}" 2>&1
  done
done

echo "Sweep complete: ${RUN_DIR}"
