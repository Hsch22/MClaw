#!/usr/bin/env bash
###############################################################################
# MClaw Full Training — Textcraft
#
# 一键启动训练（后台启 env server + 前台跑 training），日志输出到文件。
#
# 用法:
#   # 一键启动（推荐）:
#   bash examples/run_textcraft_train.sh
#
#   # 仅启动 env server:
#   bash examples/run_textcraft_train.sh env
#
#   # 仅启动训练（env server 已在另一个终端运行）:
#   bash examples/run_textcraft_train.sh train
#
# 环境变量（可覆盖）:
#   CUDA_VISIBLE_DEVICES  指定 GPU（默认 4,5）
#   NPROC_PER_NODE        训练进程数（默认 1，单卡）
#   MODEL_PATH            模型路径
#   ENV_PORT              环境服务器端口（默认 36006）
###############################################################################
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── 时间戳 ────────────────────────────────────────────────────────────────────
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# ── 路径 ──────────────────────────────────────────────────────────────────────
AGENTGYM_RL_SRC="${AGENTGYM_RL_SRC:-${HOME}/husicheng/AgentGym-RL/AgentGym-RL}"
AGENTGYM_DIR="${AGENTGYM_DIR:-${HOME}/husicheng/AgentGym-RL/AgentGym}"
TRAIN_PYTHON="${TRAIN_PYTHON:-${HOME}/husicheng/MClaw/.venv/bin/python}"
ENVSERVER_PYTHON="${ENVSERVER_PYTHON:-${HOME}/husicheng/MClaw/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-${HOME}/husicheng/AgentGym-RL/models/Qwen3-4B-Instruct-2507}"
DATA_FILE="${DATA_FILE:-${HOME}/husicheng/AgentGym-RL-Data-ID/train/textcraft_train.json}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/mclaw/config/mclaw_trainer.yaml}"

# ── GPU ───────────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

# ── 环境服务器 ─────────────────────────────────────────────────────────────────
ENV_PORT="${ENV_PORT:-36006}"
ENV_ADDR="http://127.0.0.1:${ENV_PORT}"
TASK_NAME="textcraft"

# ── 日志 ──────────────────────────────────────────────────────────────────────
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
TRAIN_LOG="${LOG_DIR}/train_${TIMESTAMP}.log"
ENV_LOG="${LOG_DIR}/env_${TIMESTAMP}.log"

# ── 输出目录 ──────────────────────────────────────────────────────────────────
OUTPUT_DIR="${PROJECT_ROOT}/outputs/${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

# ── CUDA Toolkit ──────────────────────────────────────────────────────────────
if [ -z "${CUDA_HOME:-}" ]; then
    for candidate in \
        "${HOME}/husicheng/cuda-12.4" \
        "/mnt/kangshijia/husicheng/cuda-12.4" \
        "/mnt/kangshijia/husicheng/.local/cuda-12.4"; do
        if [ -d "${candidate}" ]; then
            export CUDA_HOME="${candidate}"
            break
        fi
    done
fi
if [ -n "${CUDA_HOME:-}" ]; then
    export PATH="${CUDA_HOME}/bin:${PATH}"
    [ -d "${CUDA_HOME}/lib64" ] && export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

# ── 环境变量 ──────────────────────────────────────────────────────────────────
export TMPDIR="${TMPDIR:-/mnt/kangshijia/husicheng/tmp}"
export DS_BUILD_OPS=0
export VLLM_USE_MODELSCOPE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1
export PYTHONPATH="${AGENTGYM_RL_SRC}:${PROJECT_ROOT}:${PYTHONPATH:-}"

# ── WandB ─────────────────────────────────────────────────────────────────────
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-mclaw}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-textcraft_${TIMESTAMP}}"

# =============================================================================
#  函数
# =============================================================================

start_env_server() {
    echo "[train] Starting textcraft env server on port ${ENV_PORT}..."
    echo "[train] Env server log: ${ENV_LOG}"
    fuser -k "${ENV_PORT}/tcp" 2>/dev/null || true
    sleep 1
    cd "${AGENTGYM_DIR}/agentenv-textcraft"
    TEXTCRAFT_BIN="$(dirname "${ENVSERVER_PYTHON}")/textcraft"
    if [ -x "${TEXTCRAFT_BIN}" ]; then
        "${TEXTCRAFT_BIN}" --host 127.0.0.1 --port "${ENV_PORT}" \
            > "${ENV_LOG}" 2>&1 &
    else
        "${ENVSERVER_PYTHON}" -c "
import uvicorn
uvicorn.run('agentenv_textcraft:app', host='127.0.0.1', port=${ENV_PORT})
" > "${ENV_LOG}" 2>&1 &
    fi
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
    echo "  MClaw Full Training — Textcraft"
    echo "  Time:     ${TIMESTAMP}"
    echo "  Model:    ${MODEL_PATH}"
    echo "  Data:     ${DATA_FILE}"
    echo "  Env:      ${ENV_ADDR} (${TASK_NAME})"
    echo "  GPU:      CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    echo "  nproc:    ${NPROC_PER_NODE}"
    echo "  Output:   ${OUTPUT_DIR}"
    echo "  Log:      ${TRAIN_LOG}"
    echo "  WandB:    project=${WANDB_PROJECT}, run=${WANDB_RUN_NAME}"
    echo "================================================================"
    echo ""

    cd "${PROJECT_ROOT}"

    # 构建启动命令
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
        data.max_prompt_length=1024 \
        data.max_response_length=512 \
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
        actor_rollout_ref.rollout.max_tokens=128 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
        actor_rollout_ref.rollout.max_model_len=4096 \
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

# =============================================================================
#  入口
# =============================================================================

echo "========================================"
echo "  MClaw Training Launcher"
echo "  ${TIMESTAMP}"
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
        if [ "${1:-}" = "all" ]; then shift; fi
        run_all "$@"
        ;;
    *)
        echo "Usage: $0 {env|train|all} [extra overrides...]" >&2
        exit 1
        ;;
esac
