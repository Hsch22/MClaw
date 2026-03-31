# core

`core/` 现在承载的是“树状 rollout 本身”和 rollout 侧的本地中间表示。

## 当前文件

- `tree_node.py`
  - 只定义纯树结构 `TreeNode` 和 `resolve_action_token_weight()`。
- `contracts.py`
  - 定义 rollout 侧的批结构和协议：
  - `EnvironmentStep`
  - `AuxiliarySample`
  - `CriticSample`
  - `TrajectoryStep`
  - `TrajectoryRecord`
  - `ActorBatch`
  - `AuxiliaryBatch`
  - `CriticBatch`
  - `TreeRolloutOutput`
  - `TokenizerProtocol`
  - `InferenceEngineProtocol`
  - `EnvironmentClientProtocol`
  - `RolloutHandlerProtocol`
- `branch_selector.py`
  - 根节点 representative 选择、分支内单动作选择、簇等权辅助样本权重计算。
- `tree_rollout.py`
  - `TreeRollout`
  - `BranchRuntime`
  - 候选动作展开、Q 打分、聚类、环境执行、batch 构造。

## 已实现的行为

- root candidate 生成采用“复制同一个 prompt 组成 batch”的方式，而不是单 prompt 多返回序列。
- 后续各活跃分支的 candidate 会先合并成一个大列表做一次 `QCritic` 打分，再按父节点切片回去聚类。
- `TreeRolloutOutput` 只暴露三类 batch 和 `roots`；辅助样本、critic 样本的原始列表已经不再重复出现在输出结构里。
- `TrajectoryRecord` 已包含：
  - `response_mask`
  - `response_token_weights`
  - `advantages`
  - `returns`
  - `state_values`
  - `old_log_probs`
  - `ref_log_probs`
- observation tokens 已经正确补 0 掩码，不会再和 `input_ids` 长度错位。
- `EnvironmentClientProtocol.step()` 当前统一要求返回 `(next_state, reward, done, metadata)` 4-tuple；内部再规范化成 `EnvironmentStep`。

## 设计边界

- `contracts.py` 负责 rollout 边界类型，不负责训练后端协议；训练协议在 `trainer/contracts.py`。
- `RolloutHandlerProtocol` 仍然保留在 core 层，但当前 `TreeRollout` 的主路径主要通过 `TreeNode` 展平轨迹，而不是完全依赖 handler 对象。

## 当前状态

- core 层的 prototype 流程已经基本成形。
- 还缺真实环境适配器、真实 inference engine 输出对象和更完整的 rollout handler 实现。
