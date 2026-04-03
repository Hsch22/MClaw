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

## 用法

```bash
# 独立模式（2 GPU）
export CUDA_VISIBLE_DEVICES=2,5
bash examples/run_textcraft_train.sh

# Ray 模式（8 GPU）
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
bash examples/run_textcraft_ray_train.sh
```
