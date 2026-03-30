# core

`core/` 是树状 rollout 的核心模块，负责定义运行时数据结构和主流程接口。

## 当前文件

- `__init__.py`：导出 core 层公共类型。
- `tree_node.py`：定义纯树结构 `TreeNode` 和节点级辅助函数。
- `branch_selector.py`：定义根节点代表选择、分支内动作选择和辅助样本标记接口。
- `tree_rollout.py`：定义 `TreeRollout` 主引擎和 `BranchRuntime`。
- `contracts.py`：定义本地批结构和外部依赖协议。
  - `EnvironmentStep`
  - `AuxiliarySample`
  - `CriticSample`
  - `TrajectoryRecord`
  - `ActorBatch`
  - `AuxiliaryBatch`
  - `CriticBatch`
  - `TreeRolloutOutput`
  - `TokenizerProtocol`
  - `InferenceEngineProtocol`
  - `EnvironmentClientProtocol`
  - `RolloutHandlerProtocol`

## 关键职责

- 管理固定环境实例池
- 维护活跃分支
- 组织候选动作生成、打分、聚类和真实执行
- 构造 PPO 主样本、辅助样本和 critic 样本
- 在不直接依赖外部训练框架源码的前提下定义本地中间表示

## 当前状态

- 数据结构和协议接口已建立。
- 真实 rollout 主循环、环境执行和 batch 构造逻辑仍未实现。
