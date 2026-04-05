#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: $0 <output_root>" >&2
    exit 2
fi

OUTPUT_ROOT="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_FILE="${OUTPUT_ROOT}/orchestrator.log"

LIMIT="${LIMIT:-24}"
ROOT_BUDGET="${ROOT_BUDGET:-256}"
ROOT_CLUSTERS="${ROOT_CLUSTERS:-16}"
N_ENVS="${N_ENVS:-16}"
MAX_ROUNDS="${MAX_ROUNDS:-1}"
ROLLOUT_LOGPROBS="${ROLLOUT_LOGPROBS:-20}"
ROLLOUT_GPU_MEM="${ROLLOUT_GPU_MEM:-0.35}"
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-8192}"
TRAIN_DEVICE="${TRAIN_DEVICE:-auto}"
GPU_WAIT_ENABLED="${GPU_WAIT_ENABLED:-1}"
GPU_WAIT_MAX_STEPS="${GPU_WAIT_MAX_STEPS:-0}"
GPU_WAIT_SLEEP_SECS="${GPU_WAIT_SLEEP_SECS:-60}"
GPU_MIN_FREE_PRIMARY_MB="${GPU_MIN_FREE_PRIMARY_MB:-45000}"
GPU_MIN_FREE_SECONDARY_MB="${GPU_MIN_FREE_SECONDARY_MB:-16000}"

TASKS=(textcraft babyai maze weather)
TASK_DATA_textcraft="/mnt/kangshijia/husicheng/AgentGym-RL-Data-ID/train/textcraft_train.json"
TASK_DATA_babyai="/mnt/kangshijia/husicheng/AgentGym-RL-Data-ID/train/babyai_train.json"
TASK_DATA_maze="/mnt/kangshijia/husicheng/AgentGym-RL-Data-ID/train/lmrlgym_maze_train.json"
TASK_DATA_weather="/mnt/kangshijia/husicheng/AgentGym-RL-Data-ID/train/tool_weather_train.json"
TASK_ADDR_textcraft="http://127.0.0.1:39706"
TASK_ADDR_babyai="http://127.0.0.1:39707"
TASK_ADDR_maze="http://127.0.0.1:39716/maze"
TASK_ADDR_weather="http://127.0.0.1:39710"

VARIANT_IDS=(
    lneg1_last
    lneg2_last
    lneg4_last
    lneg8_last
    lneg1_lastk4
    lneg1_lastk8
    lneg1_mean
    lneg2_lastk4
    lneg2_lastk8
    lneg2_mean
)
VARIANT_LAYERS=(-1 -2 -4 -8 -1 -1 -1 -2 -2 -2)
VARIANT_POOLINGS=(last last last last last_k_mean last_k_mean action_mean last_k_mean last_k_mean action_mean)
VARIANT_LAST_K=(4 4 4 4 4 8 4 4 8 4)

log() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${LOG_FILE}"
}

start_env_session() {
    local session_name="$1"
    local port="$2"
    local command="$3"
    if (echo >"/dev/tcp/127.0.0.1/${port}") >/dev/null 2>&1; then
        log "env port ${port} already ready for ${session_name}"
        return 0
    fi
    if tmux has-session -t "${session_name}" 2>/dev/null; then
        log "env session ${session_name} exists but port ${port} is not ready; restarting"
        tmux kill-session -t "${session_name}"
    fi
    tmux new-session -d -s "${session_name}" "bash -lc 'cd ${PROJECT_ROOT} && ${command}'"
    log "started env session ${session_name}"
}

wait_for_exit_code() {
    local variant_id="$1"
    local task_name="$2"
    local exit_file="${OUTPUT_ROOT}/${variant_id}/${task_name}/audit.exit_code"
    while [ ! -f "${exit_file}" ]; do
        sleep 15
    done
    local status
    status="$(cat "${exit_file}")"
    log "${variant_id}/${task_name} finished with exit_code=${status}"
    if [ "${status}" -ne 0 ]; then
        return 1
    fi
}

start_variant_task() {
    local session_name="$1"
    local cuda_devices="$2"
    local variant_id="$3"
    local layer="$4"
    local pooling="$5"
    local last_k="$6"
    local task_name="$7"
    local data_file="$8"
    local env_addr="$9"
    local output_dir="${OUTPUT_ROOT}/${variant_id}/${task_name}"

    tmux new-session -d -s "${session_name}" \
        "bash -lc 'cd ${PROJECT_ROOT} && export CUDA_VISIBLE_DEVICES=${cuda_devices} LIMIT=${LIMIT} ROOT_BUDGET=${ROOT_BUDGET} ROOT_CLUSTERS=${ROOT_CLUSTERS} N_ENVS=${N_ENVS} MAX_ROUNDS=${MAX_ROUNDS} TRAIN_DEVICE=${TRAIN_DEVICE} ROLLOUT_LOGPROBS=${ROLLOUT_LOGPROBS} ROLLOUT_GPU_MEM=${ROLLOUT_GPU_MEM} ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN} GPU_WAIT_ENABLED=${GPU_WAIT_ENABLED} GPU_WAIT_MAX_STEPS=${GPU_WAIT_MAX_STEPS} GPU_WAIT_SLEEP_SECS=${GPU_WAIT_SLEEP_SECS} GPU_MIN_FREE_PRIMARY_MB=${GPU_MIN_FREE_PRIMARY_MB} GPU_MIN_FREE_SECONDARY_MB=${GPU_MIN_FREE_SECONDARY_MB} && exec bash ${PROJECT_ROOT}/tools/run_hidden_state_variant_task.sh ${task_name} ${data_file} ${env_addr} ${output_dir} ${layer} ${pooling} ${last_k}'"
    log "started ${session_name}: variant=${variant_id}, task=${task_name}, cuda=${cuda_devices}"
}

summarize_variant() {
    local variant_id="$1"
    local required_file=""
    for required_file in \
        "${OUTPUT_ROOT}/${variant_id}/textcraft/root_cluster_audit.jsonl" \
        "${OUTPUT_ROOT}/${variant_id}/babyai/root_cluster_audit.jsonl" \
        "${OUTPUT_ROOT}/${variant_id}/maze/root_cluster_audit.jsonl" \
        "${OUTPUT_ROOT}/${variant_id}/weather/root_cluster_audit.jsonl"; do
        if [ ! -f "${required_file}" ]; then
            log "skip summary for ${variant_id}: missing ${required_file}"
            return 1
        fi
    done
    python "${PROJECT_ROOT}/tools/summarize_root_audit.py" \
        --root "${OUTPUT_ROOT}/${variant_id}" \
        --tasks "textcraft,babyai,maze,weather" \
        --methods "hidden_state" \
        >> "${LOG_FILE}" 2>&1
    python "${PROJECT_ROOT}/tools/summarize_hidden_state_ablation.py" \
        --root "${OUTPUT_ROOT}" \
        >> "${LOG_FILE}" 2>&1
    log "updated summaries for ${variant_id}"
}

mkdir -p "${OUTPUT_ROOT}"

start_env_session "mclaw_hs_env_textcraft" "39706" "ENV_PORT=39706 bash examples/run_textcraft_train.sh env"
start_env_session "mclaw_hs_env_babyai" "39707" "ENV_PORT=39707 bash examples/run_babyai_train.sh env"
start_env_session "mclaw_hs_env_maze" "39716" "ENV_PORT=39716 bash examples/run_lmrlgym_maze_train.sh env"
start_env_session "mclaw_hs_env_weather" "39710" "ENV_PORT=39710 bash examples/run_tool_weather_train.sh env"

for idx in "${!VARIANT_IDS[@]}"; do
    variant_id="${VARIANT_IDS[$idx]}"
    layer="${VARIANT_LAYERS[$idx]}"
    pooling="${VARIANT_POOLINGS[$idx]}"
    last_k="${VARIANT_LAST_K[$idx]}"
    variant_root="${OUTPUT_ROOT}/${variant_id}"
    mkdir -p "${variant_root}"

    {
        echo "variant_id=${variant_id}"
        echo "layer=${layer}"
        echo "token_pooling=${pooling}"
        echo "last_k=${last_k}"
        echo "limit=${LIMIT}"
        echo "root_budget=${ROOT_BUDGET}"
        echo "root_clusters=${ROOT_CLUSTERS}"
        echo "n_envs=${N_ENVS}"
        echo "max_rounds=${MAX_ROUNDS}"
        echo "rollout_gpu_mem=${ROLLOUT_GPU_MEM}"
        echo "rollout_max_model_len=${ROLLOUT_MAX_MODEL_LEN}"
        echo "train_device=${TRAIN_DEVICE}"
        echo "gpu_wait_enabled=${GPU_WAIT_ENABLED}"
        echo "gpu_wait_max_steps=${GPU_WAIT_MAX_STEPS}"
        echo "gpu_wait_sleep_secs=${GPU_WAIT_SLEEP_SECS}"
        echo "gpu_min_free_primary_mb=${GPU_MIN_FREE_PRIMARY_MB}"
        echo "gpu_min_free_secondary_mb=${GPU_MIN_FREE_SECONDARY_MB}"
    } > "${variant_root}/variant.txt"

    log "starting variant ${variant_id} (layer=${layer}, pooling=${pooling}, last_k=${last_k})"

    start_variant_task "mclaw_hs_${variant_id}_textcraft" "2,4" "${variant_id}" "${layer}" "${pooling}" "${last_k}" textcraft "${TASK_DATA_textcraft}" "${TASK_ADDR_textcraft}"
    start_variant_task "mclaw_hs_${variant_id}_babyai" "5,7" "${variant_id}" "${layer}" "${pooling}" "${last_k}" babyai "${TASK_DATA_babyai}" "${TASK_ADDR_babyai}"
    wave1_ok=1
    if ! wait_for_exit_code "${variant_id}" textcraft; then
        wave1_ok=0
    fi
    if ! wait_for_exit_code "${variant_id}" babyai; then
        wave1_ok=0
    fi
    if [ "${wave1_ok}" -ne 1 ]; then
        log "aborting after wave1 failure in ${variant_id}"
        cc-connect send -m "hidden-state ablation 失败: ${variant_id} wave1 非零退出，详见 ${OUTPUT_ROOT}"
        exit 1
    fi

    start_variant_task "mclaw_hs_${variant_id}_maze" "2,4" "${variant_id}" "${layer}" "${pooling}" "${last_k}" maze "${TASK_DATA_maze}" "${TASK_ADDR_maze}"
    start_variant_task "mclaw_hs_${variant_id}_weather" "5,7" "${variant_id}" "${layer}" "${pooling}" "${last_k}" weather "${TASK_DATA_weather}" "${TASK_ADDR_weather}"
    wave2_ok=1
    if ! wait_for_exit_code "${variant_id}" maze; then
        wave2_ok=0
    fi
    if ! wait_for_exit_code "${variant_id}" weather; then
        wave2_ok=0
    fi
    if [ "${wave2_ok}" -ne 1 ]; then
        log "aborting after wave2 failure in ${variant_id}"
        cc-connect send -m "hidden-state ablation 失败: ${variant_id} wave2 非零退出，详见 ${OUTPUT_ROOT}"
        exit 1
    fi

    summarize_variant "${variant_id}"
done

cc-connect send -m "hidden-state ablation 已启动并完成: ${OUTPUT_ROOT}"
log "notification sent"
