#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
AGENTGYM_ROOT="${AGENTGYM_ROOT:-$(cd "${PROJECT_ROOT}/../AgentGym-RL/AgentGym-RL" && pwd)}"

CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/mclaw/config/mclaw_trainer.yaml}"
MODEL_PATH="${MODEL_PATH:-/mnt/kangshijia/husicheng/AgentGym-RL/models/Qwen3-4B-Instruct-2507}"
DATA_FILE="${DATA_FILE:-}"
ENV_ADDR="${ENV_ADDR:-http://127.0.0.1:36006}"
TASK_NAME="${TASK_NAME:-textcraft}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

export PYTHONPATH="${AGENTGYM_ROOT}:${PYTHONPATH:-}"

if [[ -z "${DATA_FILE}" ]]; then
  echo "DATA_FILE must point to a JSON/JSONL prompt dataset." >&2
  exit 1
fi

# Start the AgentEnv server separately before training. Example:
#   agentenv-server --task "${TASK_NAME}" --host 127.0.0.1 --port 36006

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" -m mclaw.trainer.main \
  --config "${CONFIG_PATH}" \
  data.train_file="${DATA_FILE}" \
  model.family="${MODEL_PATH}" \
  model.model_path="${MODEL_PATH}" \
  model.tokenizer_path="${MODEL_PATH}" \
  adapter.task_name="${TASK_NAME}" \
  adapter.env_addr="${ENV_ADDR}" \
  distributed.enable_fsdp="$([[ "${NPROC_PER_NODE}" -gt 1 ]] && echo true || echo false)" \
  distributed.tensor_parallel_size=1 \
  actor_rollout_ref.rollout.max_tokens=128 \
  "$@"
