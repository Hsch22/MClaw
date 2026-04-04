#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export MCLAW_TASK_NAME="weather"
export MCLAW_TASK_DISPLAY_NAME="Tool Weather"
export MCLAW_DATA_FILE_DEFAULT="/mnt/kangshijia/husicheng/AgentGym-RL-Data-ID/train/tool_weather_train.json"
export MCLAW_DEFAULT_ENV_PORT="36010"
export MCLAW_ENV_SUBDIR="agentenv-tool"
export MCLAW_ENV_IMPORT_MODULE="agentenv_weather"
export MCLAW_ENV_BIN_NAME="weather"
export MCLAW_ENV_PYTHONPATH_SUFFIXES="agentenv-tool"
export MCLAW_ENV_PROJECT_PATH_SUFFIX="agentenv-tool/Toolusage"
export MCLAW_ENV_LAUNCH_EXTRA_ARGS="--workers 1"
export MCLAW_ENVSERVER_PYTHON_CANDIDATES="${PROJECT_ROOT}/.venv-agentenv-tool-weather/bin/python:/home/kangshijia/miniconda3/envs/agentenv-tool/bin/python:/mnt/kangshijia/wangbinyu/conda_envs/agentenv-tool/bin/python:/mnt/kangshijia/wangbinyu/conda_envs/mclaw-envserver/bin/python"
export MCLAW_ENV_CHECK_HINT="请将 ENVSERVER_PYTHON 指向可导入 agentenv_weather 的环境；推荐先执行 \`bash install_agentenv_uv_envs.sh weather\`，或使用单独的 agentenv-tool 环境。"
export MCLAW_ENV_RUNTIME_NOTE="该环境依赖 Open-Meteo 公共 API，需要外网。adapter.task_name 使用 weather，数据文件仍是 tool_weather_train.json。"

exec "${SCRIPT_DIR}/_run_agentenv_task_train.sh" "$@"
