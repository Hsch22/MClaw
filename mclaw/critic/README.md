# critic

`critic/` 负责动作价值估计和树上的 step-level 训练信号。

## 当前文件

- `q_head.py`
  - 两层 MLP + 标量输出。
  - 输出层在初始化时做 zero init，降低初始 Q 值抖动。
  - `forward()` 现在只接受 2D `hidden_states`，不再隐式 `unsqueeze(0)`。
- `q_critic.py`
  - `score_actions()`：批量编码 `(state, action)` 并返回 `QCriticOutput`。
  - `build_td_targets()`：从 `CriticSample.metadata["next_state_value"]` 构造 TD target。
  - `update()`：重新做 backbone forward，`detach()` 后只更新 Q-head。
- `advantage.py`
  - `compute_tree_advantage()`：对 executed nodes 计算 `td_target`、`q_value`、`advantage`。
  - `estimate_state_value()`：按同一状态下候选动作的 `q_value` 均值估计 `V(s)`。

## 当前行为

- 被执行节点的真实 TD target 会覆盖 `node.q_value`，后续状态值估计优先读修正后的 `q_value`。
- `estimate_state_value()` 的优先级是：
  - 所有候选动作的 `q_value`
  - executed child 的 `td_target`
  - warning 后回退到 `0.0`
- 当节点 `done=False` 但没有 children 时，`compute_tree_advantage()` 会打 warning，并用 `V(s') = 0.0` 处理截断。
- `QCritic.update()` 里会把 backbone 暂时切到 `eval()`，并在 forward 完成后恢复原训练状态；梯度只流到 Q-head。
- `gamma` 已经收回到 `QCriticConfig`，不再在每个 `CriticSample` 上重复携带一份。

## 当前限制

- 还没有 target network、double Q 或 replay buffer。
- `build_td_targets()` 依赖 rollout 先把 `next_state_value` 放进 `CriticSample.metadata`。
- 目前的 TD 更新仍是 prototype 级 TD(0)。
