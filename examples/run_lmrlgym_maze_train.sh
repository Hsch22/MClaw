#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MCLAW_TMPDIR="${TMPDIR:-/mnt/kangshijia/husicheng/tmp}"
MCLAW_HF_CACHE_ROOT="${MCLAW_TMPDIR}/hf_lmrlgym"
mkdir -p "${MCLAW_HF_CACHE_ROOT}/hub" "${MCLAW_HF_CACHE_ROOT}/transformers"
if [ ! -s "${MCLAW_HF_CACHE_ROOT}/transformers/version.txt" ]; then
    printf '1' > "${MCLAW_HF_CACHE_ROOT}/transformers/version.txt"
fi
if [ ! -s "${MCLAW_HF_CACHE_ROOT}/hub/version.txt" ]; then
    printf '1' > "${MCLAW_HF_CACHE_ROOT}/hub/version.txt"
fi

export HF_HOME="${HF_HOME:-${MCLAW_HF_CACHE_ROOT}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${MCLAW_HF_CACHE_ROOT}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${MCLAW_HF_CACHE_ROOT}/transformers}"

export MCLAW_TASK_NAME="maze"
export MCLAW_TASK_DISPLAY_NAME="LMRL-Gym Maze"
export MCLAW_DATA_FILE_DEFAULT="/mnt/kangshijia/husicheng/AgentGym-RL-Data-ID/train/lmrlgym_maze_train.json"
export MCLAW_DEFAULT_ENV_PORT="36016"
export MCLAW_ENV_ADDR_SUFFIX="/maze"
export MCLAW_ENV_SUBDIR="agentenv-lmrlgym"
export MCLAW_ENV_IMPORT_MODULE="agentenv_lmrlgym"
export MCLAW_ENV_BIN_NAME="lmrlgym"
export MCLAW_ENV_PYTHONPATH_SUFFIXES="agentenv-lmrlgym:agentenv-lmrlgym/lmrlgym"
export MCLAW_ENVSERVER_PYTHON_CANDIDATES="${PROJECT_ROOT}/.venv-agentenv-lmrlgym/bin/python:/home/kangshijia/miniconda3/envs/agentenv-lmrlgym/bin/python:/mnt/kangshijia/wangbinyu/conda_envs/agentenv-lmrlgym/bin/python:/mnt/kangshijia/wangbinyu/conda_envs/mclaw-envserver/bin/python"
export MCLAW_ENV_CHECK_HINT="请将 ENVSERVER_PYTHON 指向可导入 agentenv_lmrlgym 的环境；推荐先执行 \`bash install_agentenv_uv_envs.sh lmrlgym\`，或使用单独的 agentenv-lmrlgym 环境。"

exec "${SCRIPT_DIR}/_run_agentenv_task_train.sh" "$@"
