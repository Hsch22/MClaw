# critic

`critic/` 负责动作价值估计和 step-level 训练信号接口。

## 当前文件

- `__init__.py`：导出 critic 层公共接口。
- `q_head.py`：定义轻量 `QHead` 模块骨架。
- `q_critic.py`：定义 `QCritic` 和 `QCriticOutput`。
- `advantage.py`：定义 `compute_tree_advantage()`、`estimate_state_value()` 和 auxiliary target 传播接口。

## 关键职责

- 计算 `Q(s, a)`
- 估计 `V(s)`
- 计算 `advantage = Q - V`
- 为被执行动作构造 TD target
- 为同簇未执行动作提供辅助监督信号

## 当前状态

- 接口、返回结构和更新入口已固定。
- 具体前向、TD target 构造和训练逻辑尚未实现。
