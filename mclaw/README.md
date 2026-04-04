# mclaw

Conda、`CUDA_HOME` / `nvcc`、flash-attn 与 vLLM 环境说明见上级目录 [README.md](../README.md)。

`mclaw/` 是仓库的主包目录。当前它不再只是 prototype 接口集合，而是已经具备一条可装配的本地训练链路：

- 配置加载
- Tree rollout
- 聚类
- Q-head critic
- verl / vLLM / agentenv 适配
- trainer 主循环
- checkpoint 保存/恢复

## 子模块

- `config/`
  - 强类型 dataclass 配置和默认 YAML
- `core/`
  - 树节点、rollout 批结构、协议、`TreeRollout`
- `adapters/`
  - `ActorBatch <-> DataProto` 适配
  - actor / ref / env / vLLM / rollout handler / logger 包装
- `clustering/`
  - action / hidden-state / output-grad / logprob / logit-distribution 聚类器
- `critic/`
  - `QHead`
  - `QCritic`
  - tree advantage / TD target 计算
- `trainer/`
  - `MClawTrainer`
  - CLI 入口
  - trainer/backend 协议
- `utils/`
  - vLLM top-k logprob、embedding matrix 等工具

## 当前包级导出

`mclaw/__init__.py` 当前导出：

- 配置：
  - `DEFAULT_CONFIG_PATH`
  - `TreeRolloutConfig`
  - `ClusteringConfig`
  - `QCriticConfig`
  - `AuxLossConfig`
  - `AdapterConfig`
  - `ModelConfig`
  - `DistributedConfig`
  - `DataConfig`
  - `TrainerRuntimeConfig`
  - `EnvironmentConfig`
  - `LoggingConfig`
  - `MClawTrainerConfig`
- core：
  - `TreeNode`
  - `TreeRollout`
  - `TreeRolloutOutput`
  - `TrajectoryRecord`
  - `TrajectoryStep`
  - `ActorBatch`
  - `AuxiliaryBatch`
  - `CriticBatch`
- critic：
  - `QHead`
  - `QCritic`
  - `compute_tree_advantage`

## 当前状态

- `trainer/main.py` 已实现：
  - OmegaConf YAML 加载
  - CLI overrides 合并
  - trainer 组装
  - `trainer.fit()` 入口
- `trainer/mclaw_trainer.py` 已实现：
  - dataloader
  - checkpoint 保存/恢复
  - 多 epoch PPO 更新
  - rollout/training sharding context 切换
- 当前仍需依赖外部运行环境做联调：
  - verl
  - vLLM
  - agentenv
  - 已初始化的 distributed/FSDP 环境
