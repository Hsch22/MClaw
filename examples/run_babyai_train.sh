#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export MCLAW_TASK_NAME="babyai"
export MCLAW_TASK_DISPLAY_NAME="BabyAI"
export MCLAW_DATA_FILE_DEFAULT="/mnt/kangshijia/husicheng/AgentGym-RL-Data-ID/train/babyai_train.json"
export MCLAW_DEFAULT_ENV_PORT="36007"
export MCLAW_ENV_SUBDIR="agentenv-babyai"
export MCLAW_ENV_IMPORT_MODULE="agentenv_babyai"
export MCLAW_ENV_BIN_NAME="babyai"
export MCLAW_ENV_PYTHONPATH_SUFFIXES="agentenv-babyai"
export MCLAW_ENVSERVER_PYTHON_CANDIDATES="${PROJECT_ROOT}/.venv-agentenv-babyai/bin/python:/home/kangshijia/miniconda3/envs/agentenv-babyai/bin/python:/mnt/kangshijia/wangbinyu/conda_envs/agentenv-babyai/bin/python:/mnt/kangshijia/wangbinyu/conda_envs/mclaw-envserver/bin/python"
export MCLAW_ENV_CHECK_HINT="请将 ENVSERVER_PYTHON 指向可导入 agentenv_babyai 的环境；推荐先执行 \`bash install_agentenv_uv_envs.sh babyai\`，或使用单独的 agentenv-babyai 环境。"

exec "${SCRIPT_DIR}/_run_agentenv_task_train.sh" "$@"
