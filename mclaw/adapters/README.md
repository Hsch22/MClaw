# adapters

`adapters/` 负责把 MClaw 的本地协议接到外部训练栈。

## 当前文件

- `dataproto_adapter.py`
  - `ActorBatch -> verl.DataProto`
  - `AdaptedActorBatch`
    - 同时保留源 `ActorBatch` 和已构造的 `DataProto`
  - continuation log-prob -> full-sequence log-prob 回写
  - 字段对齐：
    - `input_ids` / `attention_mask` / `position_ids`
    - `responses`
    - `response_mask`
    - `advantages`
    - `old_log_probs`
    - `ref_log_prob`
- `actor_backend.py`
  - `VerlActorBackend`
  - 包装 `DataParallelPPOActor` 风格后端
  - 调 `compute_log_prob()` / `update_policy()`
  - 把 verl list-metrics 规约成标量
- `ref_policy.py`
  - `VerlReferencePolicy`
  - 包装 frozen reference policy 的 `compute_ref_log_prob()`
- `env_client.py`
  - `AgentEnvClientAdapter`
  - 把 agentenv / AgentGym-RL EnvClient 统一转成 gym 风格 `step -> (next_state, reward, done, metadata)`
- `inference_engine.py`
  - `VerlInferenceEngine`
  - 包装 `llm.generate(prompts=None, prompt_token_ids=..., sampling_params=..., use_tqdm=False)`
  - 如果不直接传 `sampling_params`，则至少要求 `sampling_kwargs["max_tokens"]`
- `rollout_handler.py`
  - `VerlRolloutHandler`
  - 维护 chat messages、token 序列和 `loss_mask`
  - 可导出 `TrajectoryRecord`
- `logger.py`
  - `StandardLogger`
  - 同时兼容 verl tracking 和标准 `logging`

## 设计约束

- `responses` 使用 `prompt_length` 之后的整段 continuation，而不是
  `TrajectoryRecord.responses`
  - 原因：当前 `TrajectoryRecord.responses` 只保留 action token，不包含 observation token
  - verl 的 actor log-prob / PPO 侧需要的是完整 continuation，再用 `response_mask` 标出真正参与 loss 的位置
- MClaw 内部保存的是 full-sequence `old_log_probs` / `ref_log_probs`
  - 适配层会把 verl 返回的 continuation 张量扩展回 full sequence，再写回 `TrajectoryRecord`
- trainer 现在会先构造 `AdaptedActorBatch`
  - backend 读取其中的 `DataProto`
  - 同时保留对源 `ActorBatch` 的引用，避免 log-prob 写回链路丢失
- 适配层本身不做 Ray worker / sharding manager 调度
  - 这些仍由 `trainer/` 的训练循环负责
