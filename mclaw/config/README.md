# config

`config/` 负责定义 MClaw 的强类型配置，并提供默认 YAML。

## dataclass 配置

`config/__init__.py` 当前定义：

- `TreeRolloutConfig`
  - 根节点/分支候选预算、聚类数、最大轮数
- `ClusteringConfig`
  - 聚类方法、PCA 维度，以及 `hidden_state` / `output_grad` / `logit_distribution` 子配置
- `QCriticConfig`
  - Q-head hidden dim、MLP dim、学习率、`gamma`、更新频率、grad clip
- `AuxLossConfig`
  - auxiliary loss 系数与权重策略
- `AdapterConfig`
  - actor/ref/inference/env/logger 适配器类型，以及 `task_name`、`env_addr`、`max_retries`
- `ModelConfig`
  - `family`、`model_path`、`tokenizer_path`、`dtype`、`trust_remote_code`
- `DistributedConfig`
  - `enable_fsdp`、`tensor_parallel_size`、`fsdp_config`、rollout/training sharding manager 开关
- `DataConfig`
  - `train_file`、batch size、worker 数、prompt/response 长度、是否走 verl dataset
- `TrainerRuntimeConfig`
  - `total_epochs`、`max_steps`、`save_freq`、checkpoint 路径、resume 路径
- `EnvironmentConfig`
  - 环境适配器名、实例池大小、reset/step kwargs
- `LoggingConfig`
  - 日志路径模板、等级、project/experiment 名
- `MClawTrainerConfig`
  - 聚合上述配置，并保留 `algorithm`、`actor_rollout_ref` 外部配置树

## 默认 YAML

`mclaw_trainer.yaml` 当前分成两层：

- `mclaw.*`
  - MClaw 内部模块配置：`tree_rollout`、`clustering`、`q_critic`、`aux_loss`
- 顶层外部集成配置
  - `algorithm`
  - `actor_rollout_ref`
  - `data`
  - `trainer`
  - `environment`
  - `adapter`
  - `model`
  - `distributed`
  - `logging`

`trainer/main.py:load_config()` 会把 `mclaw.*` flatten 回 `MClawTrainerConfig` 顶层字段，因此原来的命令行覆盖方式仍然可用：

```bash
mclaw.trainer.main --config mclaw/config/mclaw_trainer.yaml \
  mclaw.tree_rollout.max_rounds=8 \
  actor_rollout_ref.actor.ppo_epochs=2
```

## 当前状态

- 默认 YAML、强类型 dataclass 和 CLI loader 现在已经对齐。
- `trainer_config_from_mapping()` 负责把普通 mapping 转成 `MClawTrainerConfig`。
- `actor_rollout_ref` 仍然保留为宽松 dict，因为 verl 的原始配置树较深，当前不适合完全 dataclass 化。
