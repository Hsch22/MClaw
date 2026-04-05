#!/usr/bin/env bash
###############################################################################
# MClaw Full Training — AgentGym Task Launcher
#
# 由任务包装脚本设置任务专属变量，再复用这一份公共训练/启动逻辑。
###############################################################################
set -euo pipefail

first_existing_executable() {
    local candidate
    for candidate in "$@"; do
        if [ -n "${candidate}" ] && [ -x "${candidate}" ]; then
            echo "${candidate}"
            return 0
        fi
    done
    return 1
}

resolve_python() {
    local resolved
    if resolved="$(first_existing_executable "$@")"; then
        echo "${resolved}"
        return 0
    fi
    command -v python
}

prepend_path_list() {
    local prefix="${1:-}"
    local suffix="${2:-}"
    if [ -n "${prefix}" ] && [ -n "${suffix}" ]; then
        echo "${prefix}:${suffix}"
    elif [ -n "${prefix}" ]; then
        echo "${prefix}"
    else
        echo "${suffix}"
    fi
}

join_agentgym_paths() {
    local raw_suffixes="${1:-}"
    local old_ifs="${IFS}"
    local suffix
    local joined=""
    local parts=()
    IFS=':' read -r -a parts <<< "${raw_suffixes}"
    IFS="${old_ifs}"
    for suffix in "${parts[@]}"; do
        if [ -z "${suffix}" ]; then
            continue
        fi
        if [ -z "${joined}" ]; then
            joined="${AGENTGYM_DIR}/${suffix}"
        else
            joined="${joined}:${AGENTGYM_DIR}/${suffix}"
        fi
    done
    echo "${joined}"
}

require_task_var() {
    local name="${1}"
    if [ -z "${!name:-}" ]; then
        echo "[train] ERROR: missing required task variable ${name}" >&2
        exit 1
    fi
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

require_task_var "MCLAW_TASK_NAME"
require_task_var "MCLAW_TASK_DISPLAY_NAME"
require_task_var "MCLAW_DATA_FILE_DEFAULT"
require_task_var "MCLAW_DEFAULT_ENV_PORT"
require_task_var "MCLAW_ENV_SUBDIR"
require_task_var "MCLAW_ENV_IMPORT_MODULE"
require_task_var "MCLAW_ENV_BIN_NAME"
require_task_var "MCLAW_ENV_PYTHONPATH_SUFFIXES"
require_task_var "MCLAW_ENV_CHECK_HINT"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

AGENTGYM_RL_SRC="${AGENTGYM_RL_SRC:-/mnt/kangshijia/husicheng/AgentGym-RL/AgentGym-RL}"
AGENTGYM_DIR="${AGENTGYM_DIR:-/mnt/kangshijia/husicheng/AgentGym}"
MODEL_PATH="${MODEL_PATH:-/mnt/kangshijia/husicheng/AgentGym-RL/models/Qwen3-4B-Instruct-2507}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/mclaw/config/mclaw_trainer.yaml}"

TRAIN_PYTHON="${TRAIN_PYTHON:-$(resolve_python \
    /mnt/kangshijia/wangbinyu/conda_envs/mclaw-train/bin/python \
    /home/kangshijia/miniconda3/envs/mclaw-train/bin/python \
    "${PROJECT_ROOT}/.venv/bin/python")}"

ENVSERVER_CANDIDATES_RAW="${MCLAW_ENVSERVER_PYTHON_CANDIDATES:-/mnt/kangshijia/wangbinyu/conda_envs/mclaw-envserver/bin/python:${PROJECT_ROOT}/.venv/bin/python}"
OLD_IFS="${IFS}"
ENVSERVER_CANDIDATES=()
IFS=':' read -r -a ENVSERVER_CANDIDATES <<< "${ENVSERVER_CANDIDATES_RAW}"
IFS="${OLD_IFS}"
ENVSERVER_PYTHON="${ENVSERVER_PYTHON:-$(resolve_python "${ENVSERVER_CANDIDATES[@]}")}"

DATA_FILE="${DATA_FILE:-${MCLAW_DATA_FILE_DEFAULT}}"
ENV_PORT="${ENV_PORT:-${MCLAW_DEFAULT_ENV_PORT}}"
ENV_ADDR="${ENV_ADDR:-http://127.0.0.1:${ENV_PORT}${MCLAW_ENV_ADDR_SUFFIX:-}}"
TASK_NAME="${MCLAW_TASK_NAME}"
DISPLAY_NAME="${MCLAW_TASK_DISPLAY_NAME}"
ENV_WORKDIR="${AGENTGYM_DIR}/${MCLAW_ENV_SUBDIR}"
ENV_PYTHONPATH_PREFIX="$(join_agentgym_paths "${MCLAW_ENV_PYTHONPATH_SUFFIXES}")"
ENV_PROJECT_PATH=""
if [ -n "${MCLAW_ENV_PROJECT_PATH_SUFFIX:-}" ]; then
    ENV_PROJECT_PATH="${AGENTGYM_DIR}/${MCLAW_ENV_PROJECT_PATH_SUFFIX}"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5,6}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
IFS=',' read -r -a _GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#_GPU_LIST[@]}"
if [ "${NUM_GPUS}" -ge 2 ]; then
    ROLLOUT_GPU_MEM="${ROLLOUT_GPU_MEM:-0.7}"
else
    ROLLOUT_GPU_MEM="${ROLLOUT_GPU_MEM:-0.4}"
fi
echo "[train] Detected ${NUM_GPUS} GPU(s), vLLM gpu_memory_utilization=${ROLLOUT_GPU_MEM}"

LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
TRAIN_LOG="${LOG_DIR}/${TASK_NAME}_train_${TIMESTAMP}.log"
ENV_LOG="${LOG_DIR}/${TASK_NAME}_env_${TIMESTAMP}.log"

OUTPUT_DIR="${PROJECT_ROOT}/outputs/${TASK_NAME}_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

if [ -z "${CUDA_HOME:-}" ]; then
    for candidate in \
        "/mnt/kangshijia/husicheng/.local/cuda-12.4" \
        "/mnt/kangshijia/husicheng/cuda-12.4" \
        "${HOME}/husicheng/cuda-12.4"; do
        if [ -d "${candidate}" ]; then
            export CUDA_HOME="${candidate}"
            break
        fi
    done
fi
if [ -n "${CUDA_HOME:-}" ]; then
    export PATH="${CUDA_HOME}/bin:${PATH}"
    if [ -d "${CUDA_HOME}/lib64" ]; then
        export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
    fi
fi

export TMPDIR="${TMPDIR:-/mnt/kangshijia/husicheng/tmp}"
export DS_BUILD_OPS=0
export VLLM_USE_MODELSCOPE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1
export PYTHONPATH="$(prepend_path_list "${AGENTGYM_RL_SRC}:${PROJECT_ROOT}" "${PYTHONPATH:-}")"

export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-mclaw}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${TASK_NAME}_${TIMESTAMP}}"

check_env_runtime() {
    if ! (
        cd "${ENV_WORKDIR}"
        export PYTHONPATH="$(prepend_path_list "${ENV_PYTHONPATH_PREFIX}" "${PYTHONPATH:-}")"
        if [ -n "${ENV_PROJECT_PATH}" ]; then
            export PROJECT_PATH="${ENV_PROJECT_PATH}"
        fi
        "${ENVSERVER_PYTHON}" -c "import ${MCLAW_ENV_IMPORT_MODULE}"
    ) >/dev/null 2>&1; then
        echo "[train] ERROR: cannot import ${MCLAW_ENV_IMPORT_MODULE} with ${ENVSERVER_PYTHON}" >&2
        echo "[train] Hint: ${MCLAW_ENV_CHECK_HINT}" >&2
        exit 1
    fi
}

start_env_server() {
    echo "[train] Starting ${DISPLAY_NAME} env server on port ${ENV_PORT}..."
    echo "[train] Env server log: ${ENV_LOG}"
    fuser -k "${ENV_PORT}/tcp" 2>/dev/null || true
    sleep 1
    check_env_runtime
    (
        cd "${ENV_WORKDIR}"
        export PYTHONPATH="$(prepend_path_list "${ENV_PYTHONPATH_PREFIX}" "${PYTHONPATH:-}")"
        if [ -n "${ENV_PROJECT_PATH}" ]; then
            export PROJECT_PATH="${ENV_PROJECT_PATH}"
        fi
        exec "${ENVSERVER_PYTHON}" -c "import uvicorn; uvicorn.run('${MCLAW_ENV_IMPORT_MODULE}:app', host='127.0.0.1', port=${ENV_PORT})"
    ) > "${ENV_LOG}" 2>&1 &
    ENV_PID=$!
    echo "[train] Env server PID: ${ENV_PID}"
}

wait_for_env_server() {
    echo "[train] Waiting for env server on port ${ENV_PORT}..."
    local max_wait=120
    local waited=0
    while ! ss -lnt 2>/dev/null | grep -q ":${ENV_PORT}" ; do
        sleep 2
        waited=$((waited + 2))
        if [ "${waited}" -ge "${max_wait}" ]; then
            echo "[train] ERROR: env server not ready after ${max_wait}s" >&2
            exit 1
        fi
    done
    echo "[train] Env server ready (waited ${waited}s)."
}

run_train() {
    wait_for_env_server

    echo ""
    echo "================================================================"
    echo "  MClaw Full Training — ${DISPLAY_NAME}"
    echo "  Time:     ${TIMESTAMP}"
    echo "  Model:    ${MODEL_PATH}"
    echo "  Data:     ${DATA_FILE}"
    echo "  Env:      ${ENV_ADDR} (${TASK_NAME})"
    echo "  GPU:      CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    echo "  nproc:    ${NPROC_PER_NODE}"
    echo "  Output:   ${OUTPUT_DIR}"
    echo "  Log:      ${TRAIN_LOG}"
    echo "  WandB:    project=${WANDB_PROJECT}, run=${WANDB_RUN_NAME}"
    if [ -n "${MCLAW_ENV_RUNTIME_NOTE:-}" ]; then
        echo "  Note:     ${MCLAW_ENV_RUNTIME_NOTE}"
    fi
    echo "================================================================"
    echo ""

    cd "${PROJECT_ROOT}"

    if [ "${NPROC_PER_NODE}" -gt 1 ]; then
        LAUNCH_CMD="torchrun --standalone --nproc_per_node=${NPROC_PER_NODE}"
    else
        LAUNCH_CMD="${TRAIN_PYTHON}"
    fi

    ${LAUNCH_CMD} -m mclaw.trainer.main \
        --config "${CONFIG_PATH}" \
        \
        data.train_file="${DATA_FILE}" \
        data.train_batch_size=1 \
        data.max_prompt_length=512 \
        data.max_response_length=10240 \
        \
        model.family="${MODEL_PATH}" \
        model.model_path="${MODEL_PATH}" \
        model.tokenizer_path="${MODEL_PATH}" \
        model.dtype=bfloat16 \
        \
        adapter.task_name="${TASK_NAME}" \
        adapter.env_addr="${ENV_ADDR}" \
        \
        distributed.enable_fsdp="$([[ "${NPROC_PER_NODE}" -gt 1 ]] && echo true || echo false)" \
        distributed.tensor_parallel_size=1 \
        \
        trainer.total_epochs=1 \
        trainer.max_steps=0 \
        trainer.save_freq=100 \
        trainer.default_local_dir="${OUTPUT_DIR}" \
        trainer.seed=42 \
        \
        mclaw.tree_rollout.root_budget=256 \
        mclaw.tree_rollout.n_envs=16 \
        mclaw.tree_rollout.root_clusters=16 \
        mclaw.tree_rollout.branch_budget=16 \
        mclaw.tree_rollout.intra_branch_clusters=4 \
        mclaw.tree_rollout.max_rounds=30 \
        \
        mclaw.clustering.method=hidden_state \
        mclaw.clustering.pca_dim=128 \
        \
        mclaw.q_critic.lr=1e-4 \
        mclaw.q_critic.gamma=0.99 \
        \
        mclaw.aux_loss.coef=0.2 \
        \
        actor_rollout_ref.rollout.max_tokens=512 \
        actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM}" \
        actor_rollout_ref.rollout.max_model_len=32768 \
        actor_rollout_ref.rollout.temperature=1.0 \
        actor_rollout_ref.rollout.n=1 \
        actor_rollout_ref.rollout.logprobs=1 \
        \
        actor_rollout_ref.actor.ppo_epochs=1 \
        actor_rollout_ref.actor.use_kl_loss=true \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.grad_clip=1.0 \
        \
        logging.tracker=wandb \
        logging.project_name="${WANDB_PROJECT}" \
        logging.experiment_name="${WANDB_RUN_NAME}" \
        logging.level=INFO \
        \
        "$@" \
        2>&1 | tee "${TRAIN_LOG}"
}

run_all() {
    start_env_server

    cleanup() {
        echo ""
        echo "[train] Cleaning up env server (PID=${ENV_PID})..."
        kill "${ENV_PID}" 2>/dev/null || true
        wait "${ENV_PID}" 2>/dev/null || true
        fuser -k "${ENV_PORT}/tcp" 2>/dev/null || true
    }
    trap cleanup EXIT INT TERM

    run_train "$@"
}

echo "========================================"
echo "  MClaw Training Launcher"
echo "  ${DISPLAY_NAME} @ ${TIMESTAMP}"
echo "========================================"

case "${1:-all}" in
    env|envserver|server)
        start_env_server
        wait "${ENV_PID}"
        ;;
    train)
        shift
        run_train "$@"
        ;;
    all)
        if [ "${1:-}" = "all" ]; then
            shift
        fi
        run_all "$@"
        ;;
    *)
        echo "Usage: $0 {env|train|all} [extra overrides...]" >&2
        exit 1
        ;;
esac
