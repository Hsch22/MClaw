#!/usr/bin/env bash
###############################################################################
# MClaw + AgentGym-RL Ray Training — Textcraft
#
# Launches MClaw's tree-search rollout integrated into AgentGym-RL's
# Ray-based distributed training framework.
#
# 用法:
#   bash examples/run_textcraft_ray_train.sh
#
# 环境变量（可覆盖）:
#   CUDA_VISIBLE_DEVICES  指定 GPU（默认 0,1,2,3,4,5,6,7）
#   N_GPUS_PER_NODE       每节点 GPU 数（默认自动检测）
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
MODEL_PATH="${MODEL_PATH:-${HOME}/husicheng/AgentGym-RL/models/Qwen3-4B-Instruct-2507}"
DATA_FILE="${DATA_FILE:-${HOME}/husicheng/AgentGym-RL-Data-ID/train/textcraft_train.json}"

# ── GPU ───────────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra _GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-${#_GPU_LIST[@]}}"

# ── 环境服务器 ─────────────────────────────────────────────────────────────────
ENV_PORT="${ENV_PORT:-36006}"
ENV_ADDR="http://127.0.0.1:${ENV_PORT}"
TASK_NAME="textcraft"

# ── 日志 ──────────────────────────────────────────────────────────────────────
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
TRAIN_LOG="${LOG_DIR}/ray_train_${TIMESTAMP}.log"
ENV_LOG="${LOG_DIR}/env_${TIMESTAMP}.log"

# ── 输出目录 ──────────────────────────────────────────────────────────────────
OUTPUT_DIR="${PROJECT_ROOT}/outputs/ray_${TIMESTAMP}"
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
WANDB_PROJECT="${WANDB_PROJECT:-mclaw-ray}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-textcraft_ray_${TIMESTAMP}}"

# ── MClaw 超参 ────────────────────────────────────────────────────────────────
ROOT_BUDGET="${ROOT_BUDGET:-256}"
N_ENVS="${N_ENVS:-16}"
ROOT_CLUSTERS="${ROOT_CLUSTERS:-16}"
BRANCH_BUDGET="${BRANCH_BUDGET:-16}"
INTRA_BRANCH_CLUSTERS="${INTRA_BRANCH_CLUSTERS:-4}"
MAX_ROUNDS="${MAX_ROUNDS:-30}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"

# =============================================================================
#  环境服务器
# =============================================================================

start_env_server() {
    echo "[train] Starting textcraft env server on port ${ENV_PORT}..."
    echo "[train] Env server log: ${ENV_LOG}"
    fuser -k "${ENV_PORT}/tcp" 2>/dev/null || true
    sleep 1
    cd "${AGENTGYM_DIR}/agentenv-textcraft"
    TEXTCRAFT_BIN="$(dirname "${TRAIN_PYTHON}")/textcraft"
    if [ -x "${TEXTCRAFT_BIN}" ]; then
        "${TEXTCRAFT_BIN}" --host 127.0.0.1 --port "${ENV_PORT}" \
            > "${ENV_LOG}" 2>&1 &
    else
        "${TRAIN_PYTHON}" -c "
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

# =============================================================================
#  训练
# =============================================================================

run_train() {
    wait_for_env_server

    echo ""
    echo "================================================================"
    echo "  MClaw Ray Training — Textcraft"
    echo "  Time:         ${TIMESTAMP}"
    echo "  Model:        ${MODEL_PATH}"
    echo "  Data:         ${DATA_FILE}"
    echo "  Env:          ${ENV_ADDR} (${TASK_NAME})"
    echo "  GPU:          CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    echo "  n_gpus:       ${N_GPUS_PER_NODE}"
    echo "  batch_size:   ${TRAIN_BATCH_SIZE}"
    echo "  Output:       ${OUTPUT_DIR}"
    echo "  Log:          ${TRAIN_LOG}"
    echo "  WandB:        project=${WANDB_PROJECT}, run=${WANDB_RUN_NAME}"
    echo "  MClaw:        root_budget=${ROOT_BUDGET}, n_envs=${N_ENVS}"
    echo "================================================================"
    echo ""

    cd "${AGENTGYM_RL_SRC}"

    ${TRAIN_PYTHON} -m verl.agent_trainer.main_ppo \
        \
        data.train_file="${DATA_FILE}" \
        data.train_batch_size="${TRAIN_BATCH_SIZE}" \
        data.max_prompt_length=512 \
        data.max_response_length=10240 \
        data.return_raw_chat=True \
        data.prompt_key=item_id \
        \
        actor_rollout_ref.model.path="${MODEL_PATH}" \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.model.trust_remote_code=True \
        actor_rollout_ref.hybrid_engine=True \
        \
        actor_rollout_ref.actor.strategy=fsdp \
        actor_rollout_ref.actor.ppo_epochs=1 \
        actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH_SIZE}" \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.actor.use_kl_loss=true \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.grad_clip=1.0 \
        \
        actor_rollout_ref.rollout.name=mclaw \
        actor_rollout_ref.rollout.max_tokens=512 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
        actor_rollout_ref.rollout.max_model_len=32768 \
        actor_rollout_ref.rollout.temperature=1.0 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.dtype=bfloat16 \
        actor_rollout_ref.rollout.enforce_eager=True \
        actor_rollout_ref.rollout.free_cache_engine=True \
        actor_rollout_ref.rollout.load_format=dummy_dtensor \
        actor_rollout_ref.rollout.disable_log_stats=True \
        actor_rollout_ref.rollout.enable_chunked_prefill=True \
        actor_rollout_ref.rollout.n=1 \
        actor_rollout_ref.rollout.logprobs=1 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
        \
        actor_rollout_ref.agentgym.task_name="${TASK_NAME}" \
        actor_rollout_ref.agentgym.env_addr="${ENV_ADDR}" \
        actor_rollout_ref.agentgym.max_retries=3 \
        actor_rollout_ref.agentgym.max_rounds="${MAX_ROUNDS}" \
        \
        actor_rollout_ref.mclaw.tree_rollout.root_budget="${ROOT_BUDGET}" \
        actor_rollout_ref.mclaw.tree_rollout.n_envs="${N_ENVS}" \
        actor_rollout_ref.mclaw.tree_rollout.root_clusters="${ROOT_CLUSTERS}" \
        actor_rollout_ref.mclaw.tree_rollout.branch_budget="${BRANCH_BUDGET}" \
        actor_rollout_ref.mclaw.tree_rollout.intra_branch_clusters="${INTRA_BRANCH_CLUSTERS}" \
        actor_rollout_ref.mclaw.tree_rollout.max_rounds="${MAX_ROUNDS}" \
        actor_rollout_ref.mclaw.clustering.method=hidden_state \
        actor_rollout_ref.mclaw.clustering.pca_dim=128 \
        actor_rollout_ref.mclaw.q_critic.lr=1e-4 \
        actor_rollout_ref.mclaw.q_critic.gamma=0.99 \
        actor_rollout_ref.mclaw.aux_loss.coef=0.2 \
        \
        algorithm.adv_estimator=mclaw \
        algorithm.rounds_ctrl.type=fixed \
        algorithm.rounds_ctrl.rounds="${MAX_ROUNDS}" \
        \
        trainer.total_epochs=1 \
        trainer.project_name="${WANDB_PROJECT}" \
        trainer.experiment_name="${WANDB_RUN_NAME}" \
        trainer.logger="['console','wandb']" \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
        trainer.save_freq=100 \
        trainer.default_local_dir="${OUTPUT_DIR}" \
        \
        "$@" \
        2>&1 | tee "${TRAIN_LOG}"
}

# =============================================================================
#  入口
# =============================================================================

echo "========================================"
echo "  MClaw Ray Training Launcher"
echo "  ${TIMESTAMP}"
echo "========================================"

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
