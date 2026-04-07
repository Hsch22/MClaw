#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-launch}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
SELF_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
ROOT_DIR="$(cd "${PROJECT_ROOT}/.." && pwd)"

TRAIN_PYTHON="${TRAIN_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
AGENTGYM_DIR="${AGENTGYM_DIR:-${ROOT_DIR}/AgentGym}"
AGENTGYM_RL_DIR="${AGENTGYM_RL_DIR:-${ROOT_DIR}/AgentGym-RL}"
AGENTGYM_RL_SRC="${AGENTGYM_RL_SRC:-${AGENTGYM_RL_DIR}/AgentGym-RL}"
VERL_DIR="${VERL_DIR:-${ROOT_DIR}/verl}"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/AgentGym-RL-Data-ID/train}"
MODEL_PATH="${MODEL_PATH:-${AGENTGYM_RL_DIR}/models/Qwen3-4B-Instruct-2507}"
TMPDIR="${TMPDIR:-${ROOT_DIR}/tmp}"
CUDA_HOME="${CUDA_HOME:-${ROOT_DIR}/cuda-12.4}"
TEXTCRAFT_ENV_PYTHON="${TEXTCRAFT_ENV_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
BABYAI_ENV_PYTHON="${BABYAI_ENV_PYTHON:-${PROJECT_ROOT}/.venv-agentenv-babyai/bin/python}"
MAZE_ENV_PYTHON="${MAZE_ENV_PYTHON:-${PROJECT_ROOT}/.venv-agentenv-lmrlgym/bin/python}"
WEATHER_ENV_PYTHON="${WEATHER_ENV_PYTHON:-${PROJECT_ROOT}/.venv-agentenv-tool-weather/bin/python}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
OUTPUTS_DIR="${OUTPUTS_DIR:-${PROJECT_ROOT}/outputs}"

RUN_ID="${RUN_ID:-hidden_state_ablation_$(date +%Y%m%d_%H%M%S)}"
RUN_LOG_DIR="${RUN_LOG_DIR:-${LOG_DIR}/${RUN_ID}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${OUTPUTS_DIR}/${RUN_ID}}"
ORCH_LOG="${RUN_LOG_DIR}/orchestrator.log"
MANIFEST_FILE="${RUN_LOG_DIR}/run_manifest.txt"

SESSION_PREFIX="${SESSION_PREFIX:-mclaw_hs4}"
SESSION_ORCH="${SESSION_PREFIX}_orchestrator"
SESSION_ENV_TEXTCRAFT="${SESSION_PREFIX}_env_textcraft"
SESSION_ENV_BABYAI="${SESSION_PREFIX}_env_babyai"
SESSION_ENV_MAZE="${SESSION_PREFIX}_env_maze"
SESSION_ENV_WEATHER="${SESSION_PREFIX}_env_weather"

PAIR_A="${PAIR_A:-0,1}"
PAIR_B="${PAIR_B:-2,3}"

TEXTCRAFT_PORT="${TEXTCRAFT_PORT:-39706}"
BABYAI_PORT="${BABYAI_PORT:-39707}"
MAZE_PORT="${MAZE_PORT:-39716}"
WEATHER_PORT="${WEATHER_PORT:-39710}"

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
    printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

env_exports() {
    cat <<EOF
export CUDA_HOME='${CUDA_HOME}';
export PATH='${CUDA_HOME}/bin':\$PATH;
export LD_LIBRARY_PATH='${CUDA_HOME}/lib64':\${LD_LIBRARY_PATH:-};
export TRAIN_PYTHON='${TRAIN_PYTHON}';
export AGENTGYM_DIR='${AGENTGYM_DIR}';
export AGENTGYM_RL_SRC='${AGENTGYM_RL_SRC}';
export MODEL_PATH='${MODEL_PATH}';
export PYTHONPATH='${PROJECT_ROOT}:${VERL_DIR}:${AGENTGYM_RL_SRC}':\${PYTHONPATH:-};
export TMPDIR='${TMPDIR}';
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy;
export NO_PROXY='127.0.0.1,localhost';
export no_proxy='127.0.0.1,localhost';
EOF
}

ensure_dirs() {
    mkdir -p "${LOG_DIR}" "${OUTPUTS_DIR}" "${RUN_LOG_DIR}" "${OUTPUT_ROOT}" "${TMPDIR}"
}

cleanup_artifacts() {
    log "cleaning empty files and empty directories under ${LOG_DIR} and ${OUTPUTS_DIR}"
    find "${LOG_DIR}" -type f -empty -print -delete 2>/dev/null || true
    find "${OUTPUTS_DIR}" -type f -empty -print -delete 2>/dev/null || true
    find "${LOG_DIR}" -depth -type d -empty -print -delete 2>/dev/null || true
    find "${OUTPUTS_DIR}" -depth -type d -empty -print -delete 2>/dev/null || true

    if [ -d "${PROJECT_ROOT}/smoke_test_output" ]; then
        log "removing smoke test output ${PROJECT_ROOT}/smoke_test_output"
        rm -rf "${PROJECT_ROOT}/smoke_test_output"
    fi
}

write_manifest() {
    cat > "${MANIFEST_FILE}" <<EOF
project_root=${PROJECT_ROOT}
root_dir=${ROOT_DIR}
output_root=${OUTPUT_ROOT}
run_log_dir=${RUN_LOG_DIR}
train_python=${TRAIN_PYTHON}
textcraft_env_python=${TEXTCRAFT_ENV_PYTHON}
babyai_env_python=${BABYAI_ENV_PYTHON}
maze_env_python=${MAZE_ENV_PYTHON}
weather_env_python=${WEATHER_ENV_PYTHON}
agentgym_dir=${AGENTGYM_DIR}
agentgym_rl_src=${AGENTGYM_RL_SRC}
verl_dir=${VERL_DIR}
data_dir=${DATA_DIR}
model_path=${MODEL_PATH}
tmpdir=${TMPDIR}
cuda_home=${CUDA_HOME}
pair_a=${PAIR_A}
pair_b=${PAIR_B}
limit=${LIMIT}
root_budget=${ROOT_BUDGET}
root_clusters=${ROOT_CLUSTERS}
n_envs=${N_ENVS}
max_rounds=${MAX_ROUNDS}
rollout_logprobs=${ROLLOUT_LOGPROBS}
rollout_gpu_mem=${ROLLOUT_GPU_MEM}
rollout_max_model_len=${ROLLOUT_MAX_MODEL_LEN}
train_device=${TRAIN_DEVICE}
gpu_wait_enabled=${GPU_WAIT_ENABLED}
gpu_wait_max_steps=${GPU_WAIT_MAX_STEPS}
gpu_wait_sleep_secs=${GPU_WAIT_SLEEP_SECS}
gpu_min_free_primary_mb=${GPU_MIN_FREE_PRIMARY_MB}
gpu_min_free_secondary_mb=${GPU_MIN_FREE_SECONDARY_MB}
EOF
}

port_ready() {
    local port="$1"
    (echo >"/dev/tcp/127.0.0.1/${port}") >/dev/null 2>&1
}

kill_session_if_exists() {
    local session_name="$1"
    if tmux has-session -t "${session_name}" 2>/dev/null; then
        tmux kill-session -t "${session_name}"
    fi
}

start_env_session() {
    local session_name="$1"
    local port="$2"
    local script_name="$3"
    local log_file="$4"
    local env_python="$5"

    if port_ready "${port}" && tmux has-session -t "${session_name}" 2>/dev/null; then
        log "env session ${session_name} already ready on port ${port}"
        return 0
    fi

    kill_session_if_exists "${session_name}"
    tmux new-session -d -s "${session_name}" \
        "bash -lc 'set -euo pipefail; cd \"${PROJECT_ROOT}\"; $(env_exports) export ENVSERVER_PYTHON=${env_python}; export ENV_PORT=${port}; bash examples/${script_name} env 2>&1 | tee -a \"${log_file}\"'"
    log "started env session ${session_name} on port ${port}"
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
    [ "${status}" -eq 0 ]
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
    local session_log="${output_dir}/session.log"

    mkdir -p "${output_dir}"
    kill_session_if_exists "${session_name}"
    tmux new-session -d -s "${session_name}" \
        "bash -lc 'set -euo pipefail; cd \"${PROJECT_ROOT}\"; $(env_exports) export CUDA_VISIBLE_DEVICES=${cuda_devices}; export LIMIT=${LIMIT} ROOT_BUDGET=${ROOT_BUDGET} ROOT_CLUSTERS=${ROOT_CLUSTERS} N_ENVS=${N_ENVS} MAX_ROUNDS=${MAX_ROUNDS} TRAIN_DEVICE=${TRAIN_DEVICE} ROLLOUT_LOGPROBS=${ROLLOUT_LOGPROBS} ROLLOUT_GPU_MEM=${ROLLOUT_GPU_MEM} ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN} GPU_WAIT_ENABLED=${GPU_WAIT_ENABLED} GPU_WAIT_MAX_STEPS=${GPU_WAIT_MAX_STEPS} GPU_WAIT_SLEEP_SECS=${GPU_WAIT_SLEEP_SECS} GPU_MIN_FREE_PRIMARY_MB=${GPU_MIN_FREE_PRIMARY_MB} GPU_MIN_FREE_SECONDARY_MB=${GPU_MIN_FREE_SECONDARY_MB}; bash tools/run_hidden_state_variant_task.sh ${task_name} ${data_file} ${env_addr} ${output_dir} ${layer} ${pooling} ${last_k} 2>&1 | tee -a \"${session_log}\"'"
    log "started ${session_name}: variant=${variant_id}, task=${task_name}, cuda=${cuda_devices}"
}

summarize_variant() {
    local variant_id="$1"
    local variant_root="${OUTPUT_ROOT}/${variant_id}"
    "${TRAIN_PYTHON}" "${PROJECT_ROOT}/tools/summarize_root_audit.py" \
        --root "${variant_root}" \
        --tasks "textcraft,babyai,maze,weather" \
        --methods "hidden_state"
    "${TRAIN_PYTHON}" "${PROJECT_ROOT}/tools/summarize_hidden_state_ablation.py" \
        --root "${OUTPUT_ROOT}" \
        --tasks "textcraft,babyai,maze,weather"
    log "updated summaries for ${variant_id}"
}

run_orchestrate() {
    ensure_dirs
    write_manifest
    exec > >(tee -a "${ORCH_LOG}") 2>&1

    log "starting hidden-state ablation"
    log "output_root=${OUTPUT_ROOT}"
    log "pair_a=${PAIR_A}, pair_b=${PAIR_B}"
    log "textcraft_env_python=${TEXTCRAFT_ENV_PYTHON}"
    log "babyai_env_python=${BABYAI_ENV_PYTHON}"
    log "maze_env_python=${MAZE_ENV_PYTHON}"
    log "weather_env_python=${WEATHER_ENV_PYTHON}"

    local variant_id=""
    local layer=""
    local pooling=""
    local last_k=""
    local idx=0

    for idx in "${!VARIANT_IDS[@]}"; do
        variant_id="${VARIANT_IDS[$idx]}"
        layer="${VARIANT_LAYERS[$idx]}"
        pooling="${VARIANT_POOLINGS[$idx]}"
        last_k="${VARIANT_LAST_K[$idx]}"
        mkdir -p "${OUTPUT_ROOT}/${variant_id}"
        cat > "${OUTPUT_ROOT}/${variant_id}/variant.txt" <<EOF
variant_id=${variant_id}
layer=${layer}
token_pooling=${pooling}
last_k=${last_k}
limit=${LIMIT}
root_budget=${ROOT_BUDGET}
root_clusters=${ROOT_CLUSTERS}
n_envs=${N_ENVS}
max_rounds=${MAX_ROUNDS}
rollout_logprobs=${ROLLOUT_LOGPROBS}
rollout_gpu_mem=${ROLLOUT_GPU_MEM}
rollout_max_model_len=${ROLLOUT_MAX_MODEL_LEN}
train_device=${TRAIN_DEVICE}
EOF

        log "starting variant ${variant_id} (layer=${layer}, pooling=${pooling}, last_k=${last_k})"

        start_variant_task "${SESSION_PREFIX}_${variant_id}_textcraft" "${PAIR_A}" "${variant_id}" "${layer}" "${pooling}" "${last_k}" \
            textcraft "${DATA_DIR}/textcraft_train.json" "http://127.0.0.1:${TEXTCRAFT_PORT}"
        start_variant_task "${SESSION_PREFIX}_${variant_id}_babyai" "${PAIR_B}" "${variant_id}" "${layer}" "${pooling}" "${last_k}" \
            babyai "${DATA_DIR}/babyai_train.json" "http://127.0.0.1:${BABYAI_PORT}"
        if ! wait_for_exit_code "${variant_id}" textcraft; then
            log "aborting after textcraft failure in ${variant_id}"
            exit 1
        fi
        if ! wait_for_exit_code "${variant_id}" babyai; then
            log "aborting after babyai failure in ${variant_id}"
            exit 1
        fi

        start_variant_task "${SESSION_PREFIX}_${variant_id}_maze" "${PAIR_A}" "${variant_id}" "${layer}" "${pooling}" "${last_k}" \
            maze "${DATA_DIR}/lmrlgym_maze_train.json" "http://127.0.0.1:${MAZE_PORT}/maze"
        start_variant_task "${SESSION_PREFIX}_${variant_id}_weather" "${PAIR_B}" "${variant_id}" "${layer}" "${pooling}" "${last_k}" \
            weather "${DATA_DIR}/tool_weather_train.json" "http://127.0.0.1:${WEATHER_PORT}"
        if ! wait_for_exit_code "${variant_id}" maze; then
            log "aborting after maze failure in ${variant_id}"
            exit 1
        fi
        if ! wait_for_exit_code "${variant_id}" weather; then
            log "aborting after weather failure in ${variant_id}"
            exit 1
        fi

        summarize_variant "${variant_id}"
    done

    log "hidden-state ablation finished successfully"
    log "variant_summary=$(printf '%s' "${OUTPUT_ROOT}/variant_summary.md")"
}

launch_all() {
    ensure_dirs
    cleanup_artifacts
    write_manifest

    start_env_session "${SESSION_ENV_TEXTCRAFT}" "${TEXTCRAFT_PORT}" "run_textcraft_train.sh" "${RUN_LOG_DIR}/textcraft_env.log" "${TEXTCRAFT_ENV_PYTHON}"
    start_env_session "${SESSION_ENV_BABYAI}" "${BABYAI_PORT}" "run_babyai_train.sh" "${RUN_LOG_DIR}/babyai_env.log" "${BABYAI_ENV_PYTHON}"
    start_env_session "${SESSION_ENV_MAZE}" "${MAZE_PORT}" "run_lmrlgym_maze_train.sh" "${RUN_LOG_DIR}/maze_env.log" "${MAZE_ENV_PYTHON}"
    start_env_session "${SESSION_ENV_WEATHER}" "${WEATHER_PORT}" "run_tool_weather_train.sh" "${RUN_LOG_DIR}/weather_env.log" "${WEATHER_ENV_PYTHON}"

    kill_session_if_exists "${SESSION_ORCH}"
    tmux new-session -d -s "${SESSION_ORCH}" \
        "bash -lc 'cd \"${PROJECT_ROOT}\"; export PROJECT_ROOT=\"${PROJECT_ROOT}\" TRAIN_PYTHON=\"${TRAIN_PYTHON}\" TEXTCRAFT_ENV_PYTHON=\"${TEXTCRAFT_ENV_PYTHON}\" BABYAI_ENV_PYTHON=\"${BABYAI_ENV_PYTHON}\" MAZE_ENV_PYTHON=\"${MAZE_ENV_PYTHON}\" WEATHER_ENV_PYTHON=\"${WEATHER_ENV_PYTHON}\" AGENTGYM_DIR=\"${AGENTGYM_DIR}\" AGENTGYM_RL_DIR=\"${AGENTGYM_RL_DIR}\" AGENTGYM_RL_SRC=\"${AGENTGYM_RL_SRC}\" VERL_DIR=\"${VERL_DIR}\" DATA_DIR=\"${DATA_DIR}\" MODEL_PATH=\"${MODEL_PATH}\" TMPDIR=\"${TMPDIR}\" CUDA_HOME=\"${CUDA_HOME}\" LOG_DIR=\"${LOG_DIR}\" OUTPUTS_DIR=\"${OUTPUTS_DIR}\" RUN_ID=\"${RUN_ID}\" RUN_LOG_DIR=\"${RUN_LOG_DIR}\" OUTPUT_ROOT=\"${OUTPUT_ROOT}\" SESSION_PREFIX=\"${SESSION_PREFIX}\" PAIR_A=\"${PAIR_A}\" PAIR_B=\"${PAIR_B}\" TEXTCRAFT_PORT=\"${TEXTCRAFT_PORT}\" BABYAI_PORT=\"${BABYAI_PORT}\" MAZE_PORT=\"${MAZE_PORT}\" WEATHER_PORT=\"${WEATHER_PORT}\" LIMIT=\"${LIMIT}\" ROOT_BUDGET=\"${ROOT_BUDGET}\" ROOT_CLUSTERS=\"${ROOT_CLUSTERS}\" N_ENVS=\"${N_ENVS}\" MAX_ROUNDS=\"${MAX_ROUNDS}\" ROLLOUT_LOGPROBS=\"${ROLLOUT_LOGPROBS}\" ROLLOUT_GPU_MEM=\"${ROLLOUT_GPU_MEM}\" ROLLOUT_MAX_MODEL_LEN=\"${ROLLOUT_MAX_MODEL_LEN}\" TRAIN_DEVICE=\"${TRAIN_DEVICE}\" GPU_WAIT_ENABLED=\"${GPU_WAIT_ENABLED}\" GPU_WAIT_MAX_STEPS=\"${GPU_WAIT_MAX_STEPS}\" GPU_WAIT_SLEEP_SECS=\"${GPU_WAIT_SLEEP_SECS}\" GPU_MIN_FREE_PRIMARY_MB=\"${GPU_MIN_FREE_PRIMARY_MB}\" GPU_MIN_FREE_SECONDARY_MB=\"${GPU_MIN_FREE_SECONDARY_MB}\"; exec \"${SELF_PATH}\" orchestrate'"

    log "launch complete"
    log "orchestrator session: ${SESSION_ORCH}"
    log "attach with: tmux attach -t ${SESSION_ORCH}"
    log "status with: ${SELF_PATH} status"
    log "logs: ${RUN_LOG_DIR}"
    log "outputs: ${OUTPUT_ROOT}"
}

show_status() {
    tmux ls 2>/dev/null | grep "${SESSION_PREFIX}" || true
}

stop_all() {
    local session_name=""
    while read -r session_name; do
        [ -n "${session_name}" ] || continue
        tmux kill-session -t "${session_name}"
        log "killed ${session_name}"
    done < <(tmux ls 2>/dev/null | grep "${SESSION_PREFIX}" | cut -d: -f1 || true)
}

case "${MODE}" in
    launch|all)
        launch_all
        ;;
    orchestrate)
        run_orchestrate
        ;;
    clean)
        ensure_dirs
        cleanup_artifacts
        ;;
    status)
        show_status
        ;;
    stop)
        stop_all
        ;;
    *)
        echo "usage: $0 [launch|orchestrate|clean|status|stop]" >&2
        exit 2
        ;;
esac
