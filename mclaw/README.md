# mclaw

`mclaw/` 是 MClaw 的主包目录，当前已经完成接口级代码骨架。

## 当前结构

```text
mclaw/
├── __init__.py
├── README.md
├── clustering/
├── config/
├── core/
├── critic/
├── trainer/
└── utils/
```

## 子模块职责

- `config/`：配置 dataclass 和默认 YAML。
- `core/`：树状 rollout 主流程、树节点、本地批结构、选择器接口。
- `clustering/`：聚类器基类和不同候选动作特征方案接口。
- `critic/`：Q-head、Q-critic、step-level advantage 接口。
- `trainer/`：训练主循环、命令行入口、训练后端协议。
- `utils/`：vLLM 兼容和 embedding / logprob 辅助接口。

## 包级导出

`__init__.py` 当前导出：

- 配置类型：`TreeRolloutConfig`、`ClusteringConfig`、`QCriticConfig`、`MClawTrainerConfig`
- 核心类型：`TreeNode`、`TreeRollout`、`TreeRolloutOutput`
- 本地批结构：`ActorBatch`、`AuxiliaryBatch`、`CriticBatch`、`TrajectoryRecord`
- critic 接口：`QHead`、`QCritic`、`compute_tree_advantage`

## 当前状态

- 已完成包结构和模块边界设计。
- 已建立与 AgentGym-RL 风格兼容的本地接口和协议。
- 未直接依赖 AgentGym-RL 源码；后续通过适配层对接外部训练后端。
