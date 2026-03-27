# config

`config/` 负责管理 MClaw 的配置类型和默认配置文件。

## 当前文件

- `__init__.py`：定义配置 dataclass。
  - `TreeRolloutConfig`
  - `ClusteringConfig`
  - `QCriticConfig`
  - `AuxLossConfig`
  - `MClawTrainerConfig`
- `mclaw_trainer.yaml`：默认训练配置骨架。

## 覆盖范围

- 根节点 rollout 参数
- 分支内 rollout 参数
- 聚类方法与超参
- Q-critic 超参
- auxiliary loss 超参
- 环境、模型、日志等占位配置

## 当前状态

- 目前配置已经能表达主流程的模块边界。
- 具体 `load_config()`、命令行 override 合并逻辑还未实现。
