# trainer

`trainer/` 负责把 rollout、actor 更新、auxiliary loss 和 Q-head 更新串起来。

## 当前文件

- `__init__.py`：导出训练模块公共接口。
- `mclaw_trainer.py`：定义 `MClawTrainer` 主循环骨架。
- `main.py`：定义命令行入口、配置加载和 trainer 构造入口。
- `contracts.py`：定义与外部训练后端解耦的协议和适配边界。
  - `ActorBackendProtocol`
  - `ReferencePolicyProtocol`
  - `LoggerProtocol`

## 关键职责

- 调用 `TreeRollout` 生成训练样本
- 调用 actor 执行 PPO 更新
- 调用 auxiliary loss 更新
- 调用 Q-head TD 更新
- 管理训练日志、checkpoint 和外部后端适配

## 设计约束

- 与 AgentGym-RL 的训练阶段保持接口兼容
- 不直接 `import` AgentGym-RL 源码
- 通过本地批结构和适配层完成解耦

## 当前状态

- 训练阶段协议和适配入口已建立。
- `fit()`、`train_step()`、批适配和真实更新逻辑尚未实现。
