# examples

## 当前文件

- `run_textcraft_train.sh`
  - MClaw 独立训练模式（单机，无 Ray）
  - 调用 `python -m mclaw.trainer.main`
  - 自动启动 textcraft 环境服务器
  - 适合调试和小规模实验

- `run_textcraft_ray_train.sh`
  - Ray 分布式训练模式（多 GPU）
  - 调用 AgentGym-RL 的 `python -m verl.agent_trainer.main_ppo`
  - `MClawTreeRollout` 替代 `vLLMRollout`，在 Ray worker 内运行树搜索
  - 支持多节点多卡扩展
  - 关键配置：`actor_rollout_ref.rollout.name=mclaw`，`algorithm.adv_estimator=mclaw`

- `run_babyai_train.sh`
  - BabyAI 独立训练模式
  - 默认端口 `36007`
  - `adapter.task_name=babyai`
  - `adapter.env_addr=http://127.0.0.1:36007`

- `run_lmrlgym_maze_train.sh`
  - LMRL-Gym Maze 独立训练模式
  - 默认端口 `36016`
  - `adapter.task_name=maze`
  - `adapter.env_addr=http://127.0.0.1:36016/maze`

- `run_lmrlgym_wordle_train.sh`
  - LMRL-Gym Wordle 独立训练模式
  - 默认端口 `36017`
  - `adapter.task_name=wordle`
  - `adapter.env_addr=http://127.0.0.1:36017/wordle`

- `run_tool_weather_train.sh`
  - Tool Weather 独立训练模式
  - 默认端口 `36010`
  - `adapter.task_name=weather`
  - `adapter.env_addr=http://127.0.0.1:36010`
  - 依赖 Open-Meteo 外网访问

- `_run_agentenv_task_train.sh`
  - 四个任务共用的启动器
  - 统一处理 env server 拉起、日志、训练主命令
  - 不直接调用，由各任务包装脚本透传参数

- `install_agentenv_uv_envs.sh`
  - 用 `uv` 创建额外 AgentGym envserver 的独立虚拟环境
  - 默认生成：
    - `.venv-agentenv-babyai`
    - `.venv-agentenv-tool-weather`
    - `.venv-agentenv-lmrlgym`
  - 安装时使用仓库内固定版本的 freeze 文件，不走“默认拉最新”

## 用法

```bash
# 独立模式（2 GPU）
export CUDA_VISIBLE_DEVICES=2,5
bash examples/run_textcraft_train.sh

# BabyAI / Maze / Wordle / Weather
bash examples/run_babyai_train.sh
bash examples/run_lmrlgym_maze_train.sh
bash examples/run_lmrlgym_wordle_train.sh
bash examples/run_tool_weather_train.sh

# Ray 模式（8 GPU）
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
bash examples/run_textcraft_ray_train.sh
```

## 环境说明

- `run_babyai_train.sh`
  - 推荐先执行 `bash install_agentenv_uv_envs.sh babyai`
  - 脚本会优先查找 `.venv-agentenv-babyai/bin/python`

- `run_lmrlgym_maze_train.sh` / `run_lmrlgym_wordle_train.sh`
  - 推荐先执行 `bash install_agentenv_uv_envs.sh lmrlgym`
  - 脚本会优先查找 `.venv-agentenv-lmrlgym/bin/python`
  - `maze` 和 `wordle` 默认分配不同端口，避免停一个脚本把另一个任务一起带掉

- `run_tool_weather_train.sh`
  - 推荐先执行 `bash install_agentenv_uv_envs.sh weather`
  - 脚本会优先查找 `.venv-agentenv-tool-weather/bin/python`
  - 服务必须从 `agentenv-tool` 根目录启动，脚本已自动处理 `PROJECT_PATH`
