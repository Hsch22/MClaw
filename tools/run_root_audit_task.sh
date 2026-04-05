#!/usr/bin/env bash
set -uo pipefail

if [ "$#" -ne 4 ]; then
    echo "usage: $0 <task_name> <data_file> <env_addr> <output_dir>" >&2
    exit 2
fi

TASK_NAME="$1"
DATA_FILE="$2"
ENV_ADDR="$3"
OUTPUT_DIR="$4"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TRAIN_PYTHON="${TRAIN_PYTHON:-/mnt/kangshijia/wangbinyu/conda_envs/mclaw-train/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/mclaw/config/mclaw_trainer.yaml}"
MODEL_PATH="${MODEL_PATH:-/mnt/kangshijia/husicheng/AgentGym-RL/models/Qwen3-4B-Instruct-2507}"
AGENTGYM_RL_SRC="${AGENTGYM_RL_SRC:-/mnt/kangshijia/husicheng/AgentGym-RL/AgentGym-RL}"

LIMIT="${LIMIT:-50}"
METHODS="${METHODS:-action,hidden_state,output_grad,logprob,logit_distribution}"
ROOT_BUDGET="${ROOT_BUDGET:-256}"
ROOT_CLUSTERS="${ROOT_CLUSTERS:-16}"
N_ENVS="${N_ENVS:-16}"
MAX_ROUNDS="${MAX_ROUNDS:-1}"
TRAIN_DEVICE="${TRAIN_DEVICE:-auto}"
ROLLOUT_LOGPROBS="${ROLLOUT_LOGPROBS:-20}"
ROLLOUT_MAX_TOKENS="${ROLLOUT_MAX_TOKENS:-512}"
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-32768}"
ROLLOUT_GPU_MEM="${ROLLOUT_GPU_MEM:-0.45}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-10240}"
PORT_WAIT_STEPS="${PORT_WAIT_STEPS:-180}"
GPU_WAIT_ENABLED="${GPU_WAIT_ENABLED:-1}"
GPU_WAIT_MAX_STEPS="${GPU_WAIT_MAX_STEPS:-0}"
GPU_WAIT_SLEEP_SECS="${GPU_WAIT_SLEEP_SECS:-60}"
GPU_MIN_FREE_PRIMARY_MB="${GPU_MIN_FREE_PRIMARY_MB:-45000}"
GPU_MIN_FREE_SECONDARY_MB="${GPU_MIN_FREE_SECONDARY_MB:-16000}"
SEED="${SEED:-42}"

mkdir -p "${OUTPUT_DIR}"
AUDIT_LOG="${OUTPUT_DIR}/audit.log"
EXIT_CODE_FILE="${OUTPUT_DIR}/audit.exit_code"
COMMAND_FILE="${OUTPUT_DIR}/audit.command.txt"

export TMPDIR="${TMPDIR:-/mnt/kangshijia/husicheng/tmp}"
export DS_BUILD_OPS=0
export VLLM_USE_MODELSCOPE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1
export PYTHONPATH="${AGENTGYM_RL_SRC}:${PROJECT_ROOT}:${PYTHONPATH:-}"

ENV_PORT="$(printf '%s' "${ENV_ADDR}" | sed -E 's#.*:([0-9]+).*#\1#')"
VISIBLE_GPU_IDS=()
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -r -a RAW_VISIBLE_GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
    for gpu_id in "${RAW_VISIBLE_GPU_IDS[@]}"; do
        gpu_id="${gpu_id//[[:space:]]/}"
        if [[ "${gpu_id}" =~ ^[0-9]+$ ]]; then
            VISIBLE_GPU_IDS+=("${gpu_id}")
        fi
    done
fi

: > "${AUDIT_LOG}"

EXTRA_ARGS=()
if [ -n "${EXTRA_OVERRIDES:-}" ]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS=( ${EXTRA_OVERRIDES} )
fi

{
    echo "task_name=${TASK_NAME}"
    echo "data_file=${DATA_FILE}"
    echo "env_addr=${ENV_ADDR}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
    echo "limit=${LIMIT}"
    echo "methods=${METHODS}"
    echo "root_budget=${ROOT_BUDGET}"
    echo "root_clusters=${ROOT_CLUSTERS}"
    echo "n_envs=${N_ENVS}"
    echo "max_rounds=${MAX_ROUNDS}"
    echo "train_device=${TRAIN_DEVICE}"
    echo "rollout_logprobs=${ROLLOUT_LOGPROBS}"
    echo "rollout_max_tokens=${ROLLOUT_MAX_TOKENS}"
    echo "rollout_max_model_len=${ROLLOUT_MAX_MODEL_LEN}"
    echo "rollout_gpu_memory_utilization=${ROLLOUT_GPU_MEM}"
    echo "gpu_wait_enabled=${GPU_WAIT_ENABLED}"
    echo "gpu_wait_max_steps=${GPU_WAIT_MAX_STEPS}"
    echo "gpu_wait_sleep_secs=${GPU_WAIT_SLEEP_SECS}"
    echo "gpu_min_free_primary_mb=${GPU_MIN_FREE_PRIMARY_MB}"
    echo "gpu_min_free_secondary_mb=${GPU_MIN_FREE_SECONDARY_MB}"
    echo "extra_overrides=${EXTRA_OVERRIDES:-}"
} > "${COMMAND_FILE}"

log_audit() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*" >> "${AUDIT_LOG}"
}

get_gpu_free_mb() {
    local target_gpu="$1"
    nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits \
        | awk -F',' -v target="${target_gpu}" '
            {
                gsub(/ /, "", $1)
                gsub(/ /, "", $2)
                gsub(/ /, "", $3)
                if ($1 == target) {
                    print $2 - $3
                    exit
                }
            }
        '
}

wait_for_visible_gpus() {
    if [ "${GPU_WAIT_ENABLED}" != "1" ] || [ "${#VISIBLE_GPU_IDS[@]}" -eq 0 ]; then
        return 0
    fi

    local attempt=0
    while true; do
        local ready=1
        local status_parts=()
        local visible_index=0
        local gpu_id=""
        for gpu_id in "${VISIBLE_GPU_IDS[@]}"; do
            local free_mb=""
            local min_free_mb="${GPU_MIN_FREE_SECONDARY_MB}"
            if [ "${visible_index}" -eq 0 ]; then
                min_free_mb="${GPU_MIN_FREE_PRIMARY_MB}"
            fi
            free_mb="$(get_gpu_free_mb "${gpu_id}" || true)"
            if [ -z "${free_mb}" ]; then
                ready=0
                status_parts+=("gpu${gpu_id}=query_failed/need${min_free_mb}")
            elif [ "${free_mb}" -lt "${min_free_mb}" ]; then
                ready=0
                status_parts+=("gpu${gpu_id}=free${free_mb}/need${min_free_mb}")
            else
                status_parts+=("gpu${gpu_id}=free${free_mb}/need${min_free_mb}")
            fi
            visible_index=$((visible_index + 1))
        done

        if [ "${ready}" -eq 1 ]; then
            log_audit "gpu preflight passed: cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-} ${status_parts[*]}"
            return 0
        fi

        attempt=$((attempt + 1))
        if [ "${GPU_WAIT_MAX_STEPS}" -gt 0 ] && [ "${attempt}" -ge "${GPU_WAIT_MAX_STEPS}" ]; then
            log_audit "gpu preflight timed out after ${attempt} checks: cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-} ${status_parts[*]}"
            return 1
        fi
        log_audit "waiting for GPUs: cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-} ${status_parts[*]}"
        sleep "${GPU_WAIT_SLEEP_SECS}"
    done
}

port_ready() {
    (echo >"/dev/tcp/127.0.0.1/${ENV_PORT}") >/dev/null 2>&1
}

wait_for_env_port() {
    local attempt=""
    for attempt in $(seq 1 "${PORT_WAIT_STEPS}"); do
        if port_ready; then
            return 0
        fi
        sleep 2
    done
    return 1
}

if ! wait_for_env_port; then
    log_audit "env server on port ${ENV_PORT} not ready"
    printf '%s\n' "1" > "${EXIT_CODE_FILE}"
    exit 1
fi

if ! wait_for_visible_gpus; then
    printf '%s\n' "1" > "${EXIT_CODE_FILE}"
    exit 1
fi

if ! wait_for_env_port; then
    log_audit "env server on port ${ENV_PORT} disappeared before audit start"
    printf '%s\n' "1" > "${EXIT_CODE_FILE}"
    exit 1
fi

"${TRAIN_PYTHON}" "${PROJECT_ROOT}/tools/root_cluster_audit.py" \
    --config "${CONFIG_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --limit "${LIMIT}" \
    --methods "${METHODS}" \
    data.train_file="${DATA_FILE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    model.family="${MODEL_PATH}" \
    model.model_path="${MODEL_PATH}" \
    model.tokenizer_path="${MODEL_PATH}" \
    model.dtype=bfloat16 \
    adapter.task_name="${TASK_NAME}" \
    adapter.env_addr="${ENV_ADDR}" \
    distributed.enable_fsdp=false \
    distributed.tensor_parallel_size=1 \
    distributed.train_device="${TRAIN_DEVICE}" \
    trainer.seed="${SEED}" \
    logging.tracker=none \
    mclaw.tree_rollout.root_budget="${ROOT_BUDGET}" \
    mclaw.tree_rollout.n_envs="${N_ENVS}" \
    mclaw.tree_rollout.root_clusters="${ROOT_CLUSTERS}" \
    mclaw.tree_rollout.branch_budget=16 \
    mclaw.tree_rollout.intra_branch_clusters=4 \
    mclaw.tree_rollout.max_rounds="${MAX_ROUNDS}" \
    actor_rollout_ref.rollout.max_tokens="${ROLLOUT_MAX_TOKENS}" \
    actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM}" \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.logprobs="${ROLLOUT_LOGPROBS}" \
    "${EXTRA_ARGS[@]}" \
    >> "${AUDIT_LOG}" 2>&1
STATUS=$?

printf '%s\n' "${STATUS}" > "${EXIT_CODE_FILE}"
exit "${STATUS}"
