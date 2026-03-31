# trainer

`trainer/` 负责把 rollout 输出接到 actor / ref / critic 更新链路上，并提供训练入口。

## 当前文件

- `contracts.py`
  - `ActorBackendProtocol`
  - `ReferencePolicyProtocol`
  - `LoggerProtocol`
- `mclaw_trainer.py`
  - `MClawTrainer`
  - dataloader 构建
  - checkpoint 保存/恢复
  - rollout/training context 切换
- `main.py`
  - YAML/OmegaConf 加载
  - tokenizer / model / QCritic / clusterer / inference / env / logger 组装
  - CLI 主入口

## 当前训练链路

- `fit()`
  - 构建 `TreeRollout`
  - 构建 dataloader
  - 可选恢复 checkpoint
  - 按 epoch/batch 调 `train_step()`
  - 按 `save_freq` 保存 checkpoint
- `train_step()`
  - rollout
  - 重算 actor `old_log_probs`
  - 重算 reference `ref_log_probs`
  - 多 epoch PPO `update_policy()`
  - Q-head `update()`
- `save_checkpoint()`
  - 保存 backbone/FSDP state dict
  - 保存 Q-head state dict
  - 保存 actor optimizer / q_critic optimizer state
  - 保存 `global_step`、`epoch`、`batch_index`

## DataProto 适配职责

- trainer 侧 `adapt_actor_batch()`
  - 调 `DataProtoAdapter.adapt_actor_batch()`
  - 返回 `AdaptedActorBatch`
- backend 侧
  - `VerlActorBackend` / `VerlReferencePolicy` 读取其中的 `DataProto`
  - 同时保留源 `ActorBatch`，便于把 log-prob 回写到 `TrajectoryRecord`

这层拆分的目的是：

- trainer 真正负责“是否做格式适配”
- backend 负责“如何调用外部 actor/ref 接口”
- log-prob 写回不需要再从 trainer 手工 scatter 一遍

## 当前限制

- `build_trainer()` 已经能实例化本地/verl/vLLM 组件，但运行仍依赖外部环境：
  - `transformers`
  - `verl`
  - `vllm`
  - `agentenv`
- `FSDP` 只有在 `torch.distributed` 已初始化时才会启用；否则会直接报错而不是静默假装成功。
