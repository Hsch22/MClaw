#!/usr/bin/env bash
set -euo pipefail

# TextCraft 训练启动示例；实际环境路径和并行参数后续再细化。
python -m mclaw.trainer.main \
  --config mclaw/config/mclaw_trainer.yaml \
  model.family=/mnt/kangshijia/husicheng/AgentGym-RL/models/Qwen3-4B-Instruct-2507 \
  environment.adapter=agentgym-rl-qwen3 \
  "$@"
