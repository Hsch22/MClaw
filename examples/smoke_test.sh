#!/usr/bin/env bash
###############################################################################
# MClaw Smoke Test
#
# 用最小配置验证 MClaw 训练流程是否跑通（1 step, 小 budget）。
# 使用 textcraft 环境（本机已装好），单 GPU，Qwen3-4B 模型。
#
# 用法:
#   # 1. 先在另一个终端启动 textcraft 环境服务器:
#   bash examples/smoke_test.sh env
#
#   # 2. 等服务器就绪后，在另一个终端运行训练:
#   bash examples/smoke_test.sh train
#
#   # 或者一键启动（后台启服务器 + 前台训练）:
#   bash examples/smoke_test.sh all
#
# 环境变量:
#   CUDA_VISIBLE_DEVICES  指定 GPU（默认 4,5）
#   MODEL_PATH            模型路径（默认 Qwen3-4B-Instruct-2507）
#   ENV_PORT              环境服务器端口（默认 36006）
#   TRAIN_GPU             训练用 GPU（默认 CUDA_VISIBLE_DEVICES 的第一个）
###############################################################################
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── 路径 ──────────────────────────────────────────────────────────────────────
AGENTGYM_RL_SRC="${AGENTGYM_RL_SRC:-/mnt/kangshijia/husicheng/AgentGym-RL/AgentGym-RL}"
AGENTGYM_DIR="${AGENTGYM_DIR:-/mnt/kangshijia/husicheng/AgentGym}"
TRAIN_PYTHON="${TRAIN_PYTHON:-/mnt/kangshijia/wangbinyu/conda_envs/mclaw-train/bin/python}"
ENVSERVER_PYTHON="${ENVSERVER_PYTHON:-/mnt/kangshijia/wangbinyu/conda_envs/mclaw-envserver/bin/python}"
MODEL_PATH="${MODEL_PATH:-/mnt/kangshijia/husicheng/AgentGym-RL/models/Qwen3-4B-Instruct-2507}"
DATA_FILE="${DATA_FILE:-/mnt/kangshijia/husicheng/AgentGym-RL-Data-ID/train/textcraft_train.json}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/mclaw/config/mclaw_trainer.yaml}"

# ── GPU ───────────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"

# ── 环境服务器 ─────────────────────────────────────────────────────────────────
ENV_PORT="${ENV_PORT:-36006}"
ENV_ADDR="http://127.0.0.1:${ENV_PORT}"
TASK_NAME="textcraft"

# ── CUDA Toolkit ──────────────────────────────────────────────────────────────
if [ -z "${CUDA_HOME:-}" ]; then
    if [ -d "/mnt/kangshijia/husicheng/.local/cuda-12.4" ]; then
        export CUDA_HOME="/mnt/kangshijia/husicheng/.local/cuda-12.4"
    elif command -v nvcc >/dev/null 2>&1; then
        CUDA_BIN_DIR="$(dirname "$(command -v nvcc)")"
        export CUDA_HOME="$(cd "${CUDA_BIN_DIR}/.." && pwd)"
    fi
fi
if [ -n "${CUDA_HOME:-}" ]; then
    export PATH="${CUDA_HOME}/bin:${PATH}"
    [ -d "${CUDA_HOME}/lib64" ] && export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

# ── vLLM / PyTorch ────────────────────────────────────────────────────────────
export VLLM_USE_MODELSCOPE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1

# ── PYTHONPATH ────────────────────────────────────────────────────────────────
export PYTHONPATH="${AGENTGYM_RL_SRC}:${PROJECT_ROOT}:${PYTHONPATH:-}"

# =============================================================================
#  子命令
# =============================================================================

start_env_server() {
    echo "[smoke_test] Starting textcraft env server on port ${ENV_PORT}..."
    fuser -k "${ENV_PORT}/tcp" 2>/dev/null || true
    sleep 1
    cd "${AGENTGYM_DIR}/agentenv-textcraft"
    TEXTCRAFT_BIN="/mnt/kangshijia/wangbinyu/conda_envs/mclaw-envserver/bin/textcraft"
    if [ -x "${TEXTCRAFT_BIN}" ]; then
        exec "${TEXTCRAFT_BIN}" --host 127.0.0.1 --port "${ENV_PORT}"
    else
        exec "${ENVSERVER_PYTHON}" -c "
import uvicorn
uvicorn.run('agentenv_textcraft:app', host='127.0.0.1', port=${ENV_PORT})
"
    fi
}

wait_for_env_server() {
    echo "[smoke_test] Waiting for env server on port ${ENV_PORT}..."
    local max_wait=120
    local waited=0
    while ! ss -lnt 2>/dev/null | grep -q ":${ENV_PORT}" ; do
        sleep 2
        waited=$((waited + 2))
        if [ "${waited}" -ge "${max_wait}" ]; then
            echo "[smoke_test] ERROR: env server not ready after ${max_wait}s" >&2
            exit 1
        fi
    done
    echo "[smoke_test] Env server ready (waited ${waited}s)."
}

run_smoke_train() {
    wait_for_env_server

    echo ""
    echo "================================================================"
    echo "  MClaw Smoke Test — Training"
    echo "  Model:    ${MODEL_PATH}"
    echo "  Data:     ${DATA_FILE}"
    echo "  Env:      ${ENV_ADDR} (${TASK_NAME})"
    echo "  GPU:      CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    echo "================================================================"
    echo ""

    cd "${PROJECT_ROOT}"

    # 单 GPU, 不用 torchrun，直接 python -m
    "${TRAIN_PYTHON}" -m mclaw.trainer.main \
        --config "${CONFIG_PATH}" \
        data.train_file="${DATA_FILE}" \
        data.train_batch_size=1 \
        data.max_prompt_length=512 \
        data.max_response_length=256 \
        model.family="${MODEL_PATH}" \
        model.model_path="${MODEL_PATH}" \
        model.tokenizer_path="${MODEL_PATH}" \
        model.dtype=bfloat16 \
        adapter.task_name="${TASK_NAME}" \
        adapter.env_addr="${ENV_ADDR}" \
        distributed.enable_fsdp=false \
        distributed.tensor_parallel_size=1 \
        mclaw.tree_rollout.root_budget=4 \
        mclaw.tree_rollout.n_envs=2 \
        mclaw.tree_rollout.root_clusters=2 \
        mclaw.tree_rollout.branch_budget=4 \
        mclaw.tree_rollout.intra_branch_clusters=2 \
        mclaw.tree_rollout.max_rounds=2 \
        mclaw.clustering.method=logprob \
        mclaw.clustering.pca_dim=32 \
        mclaw.q_critic.lr=1e-4 \
        mclaw.q_critic.gamma=0.99 \
        mclaw.aux_loss.coef=0.1 \
        actor_rollout_ref.rollout.max_tokens=64 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
        actor_rollout_ref.rollout.max_model_len=2048 \
        actor_rollout_ref.rollout.temperature=1.0 \
        actor_rollout_ref.rollout.n=1 \
        actor_rollout_ref.rollout.logprobs=1 \
        actor_rollout_ref.actor.ppo_epochs=1 \
        actor_rollout_ref.actor.use_kl_loss=false \
        trainer.total_epochs=1 \
        trainer.max_steps=1 \
        trainer.save_freq=0 \
        trainer.default_local_dir="${PROJECT_ROOT}/smoke_test_output" \
        logging.tracker=none \
        logging.level=DEBUG \
        "$@"

    echo ""
    echo "[smoke_test] ✅ Smoke test completed successfully!"
}

run_all() {
    echo "[smoke_test] Starting env server in background..."
    start_env_server &
    env_pid=$!

    cleanup() {
        echo "[smoke_test] Cleaning up env server (PID=${env_pid})..."
        kill "${env_pid}" 2>/dev/null || true
        wait "${env_pid}" 2>/dev/null || true
        fuser -k "${ENV_PORT}/tcp" 2>/dev/null || true
    }
    trap cleanup EXIT INT TERM

    run_smoke_train "$@"
}

# =============================================================================
#  入口
# =============================================================================

case "${1:-all}" in
    env|envserver|server)
        start_env_server
        ;;
    train)
        shift
        run_smoke_train "$@"
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
