#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export MCLAW_TASK_NAME="wordle"
export MCLAW_TASK_DISPLAY_NAME="LMRL-Gym Wordle"
export MCLAW_DATA_FILE_DEFAULT="/mnt/kangshijia/husicheng/AgentGym-RL-Data-ID/train/lmrlgym_wordle_train.json"
export MCLAW_DEFAULT_ENV_PORT="36017"
export MCLAW_ENV_ADDR_SUFFIX="/wordle"
export MCLAW_ENV_SUBDIR="agentenv-lmrlgym"
export MCLAW_ENV_IMPORT_MODULE="agentenv_lmrlgym"
export MCLAW_ENV_BIN_NAME="lmrlgym"
export MCLAW_ENV_PYTHONPATH_SUFFIXES="agentenv-lmrlgym:agentenv-lmrlgym/lmrlgym"
export MCLAW_ENVSERVER_PYTHON_CANDIDATES="${PROJECT_ROOT}/.venv-agentenv-lmrlgym/bin/python:/home/kangshijia/miniconda3/envs/agentenv-lmrlgym/bin/python:/mnt/kangshijia/wangbinyu/conda_envs/agentenv-lmrlgym/bin/python:/mnt/kangshijia/wangbinyu/conda_envs/mclaw-envserver/bin/python"
export MCLAW_ENV_CHECK_HINT="请将 ENVSERVER_PYTHON 指向可导入 agentenv_lmrlgym 的环境；推荐先执行 \`bash install_agentenv_uv_envs.sh lmrlgym\`，或使用单独的 agentenv-lmrlgym 环境。"

exec "${SCRIPT_DIR}/_run_agentenv_task_train.sh" "$@"
