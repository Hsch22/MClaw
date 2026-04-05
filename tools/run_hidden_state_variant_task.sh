#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 7 ]; then
    echo "usage: $0 <task_name> <data_file> <env_addr> <output_dir> <layer> <token_pooling> <last_k>" >&2
    exit 2
fi

TASK_NAME="$1"
DATA_FILE="$2"
ENV_ADDR="$3"
OUTPUT_DIR="$4"
LAYER="$5"
TOKEN_POOLING="$6"
LAST_K="$7"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VARIANT_OVERRIDES=(
    "mclaw.clustering.method=hidden_state"
    "mclaw.clustering.hidden_state.layer=${LAYER}"
    "mclaw.clustering.hidden_state.token_pooling=${TOKEN_POOLING}"
    "mclaw.clustering.hidden_state.last_k=${LAST_K}"
)

export METHODS="hidden_state"
export EXTRA_OVERRIDES="${VARIANT_OVERRIDES[*]}"

exec bash "${SCRIPT_DIR}/run_root_audit_task.sh" \
    "${TASK_NAME}" \
    "${DATA_FILE}" \
    "${ENV_ADDR}" \
    "${OUTPUT_DIR}"
