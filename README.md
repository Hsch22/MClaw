# MClaw

环境为
agentgym-rl-qwen3-clean     /mnt/kangshijia/wangbinyu/conda_envs/agentgym-rl-qwen3-clean
agentenv-webarena        /mnt/kangshijia/wangbinyu/conda_envs/agentenv-webarena



MClaw 是一个面向 Agent RL 的树状 rollout prototype。当前代码不再只是“空骨架”: `TreeRollout`、`BaseClusterer`、`QCritic`、`compute_tree_advantage()` 和 `MClawTrainer.train_step()` 都已经有可执行的原型逻辑；仍然缺的是配置加载、外部后端适配、`fit()` 主循环和端到端联调。

## 当前实现处于什么阶段

- 已实装的原型逻辑：
  - 树状 rollout 主流程：根节点批量展开、后续分支展开、候选动作批量打分、聚类、真实执行、轨迹展平。
  - 聚类基类：可选 PCA 降维 + 确定性的 torch K-Means + representative 选择。
  - 四类聚类特征：`hidden_state`、`output_grad`、`logprob`、`logit_distribution`。
  - Q-head / Q-critic：共享 backbone 编码 `[state + action]`，只更新 Q-head。
  - 树上的 step-level advantage：被执行节点用 TD target 覆盖 `q_value`，再计算 `advantage = Q - V`。
  - trainer 单步更新：重算 actor old/ref log probs，合并 auxiliary loss 信号，支持多 epoch PPO 和多次 Q-head update。

- 仍然是 TODO 的部分：
  - `mclaw.trainer.main` 里的 `load_config()`、`build_trainer()`、`main()`。
  - `MClawTrainer.fit()`、`save_checkpoint()`、`build_rollout_engine()`。
  - actor / ref policy / env / logger 的真实适配层。
  - 和 AgentGym-RL 风格训练栈的端到端集成。

## 仓库结构

```text
MClaw/
├── README.md
├── plan.md
├── setup.py
├── examples/
│   ├── README.md
│   └── textcraft_train.sh
└── mclaw/
    ├── README.md
    ├── __init__.py
    ├── clustering/
    ├── config/
    ├── core/
    ├── critic/
    ├── trainer/
    └── utils/
```

## 模块概览

- `plan.md`
  - 算法设计文档，当前实现大体按这里的模块边界推进。

- `mclaw/config/`
  - dataclass 配置定义和默认 YAML。
  - 已导出 `DEFAULT_CONFIG_PATH`、`TreeRolloutConfig`、`ClusteringConfig`、`QCriticConfig`、`AuxLossConfig`、`MClawTrainerConfig`。

- `mclaw/core/`
  - 本地批结构和外部协议。
  - `TreeNode` 现在是纯树节点定义；训练样本容器已经移到 `contracts.py`。
  - `TreeRolloutOutput` 只保留 `actor_data`、`aux_actor_data`、`critic_data` 和 `roots`，不再重复暴露原始 sample 列表。
  - `EnvironmentClientProtocol.step()` 现在统一要求返回 gym 风格 4-tuple。

- `mclaw/clustering/`
  - `BaseClusterer` 已包含实际的特征清洗、可选 PCA、确定性 K-Means 和 representative 选择。
  - 各具体 clusterer 已有特征提取实现，而不是占位文件。

- `mclaw/critic/`
  - `QHead` 已实现两层 MLP + zero-init 输出层。
  - `QCritic` 已实现批量编码、TD target 构造和 Q-head 更新。
  - `advantage.py` 已实现 executed-node 的 TD/advantage 回填和状态值估计。

- `mclaw/trainer/`
  - `train_step()`、`update_actor()`、`update_q_head()` 已实现。
  - 训练后端协议位于 `trainer/contracts.py`。
  - CLI 入口文件存在，但配置加载和 trainer 组装仍未接好。

- `mclaw/utils/`
  - `EmbeddingMatrixCache`、`extract_topk_logprobs()`、`build_output_grad_features()` 已实现。
  - 面向 vLLM top-k logprob 输出的解析逻辑已从“猜格式”收紧为较严格的契约。

## 当前训练数据流

1. `TreeRollout.generate_tree_rollout()` 对每个 prompt 构建 root。
2. 根节点用“重复 prompt 组成 batch”的方式生成 `root_budget` 个候选动作。
3. `QCritic.score_actions()` 对所有候选 `(state, action)` 批量打分。
4. `clusterer.cluster_candidates()` 做聚类，`BranchSelector` 选代表动作。
5. 被选动作送入环境执行，生成 observation tokens 并写回节点。
6. `compute_tree_advantage()` 反向计算 executed nodes 的 `td_target`、`q_value`、`advantage`。
7. rollout 构造三类本地 batch：
   - `ActorBatch`
   - `AuxiliaryBatch`
   - `CriticBatch`
8. `MClawTrainer.train_step()` 调用：
   - `compute_old_log_probs()`
   - `compute_ref_log_probs()`
   - 多 epoch `update_policy()`
   - 多次 `q_critic.update()`

## 与 AgentGym-RL 的关系

MClaw 的目标仍然是语义兼容，而不是源码耦合。

- `core/contracts.py` 定义 rollout 侧本地批结构和协议。
- `trainer/contracts.py` 定义训练后端协议。
- `MClawTrainer` 通过 `adapt_actor_batch()` / `adapt_critic_batch()` 等钩子为外部后端预留适配入口。

因此当前仓库可以单独迭代核心逻辑，接入 AgentGym-RL 时只需要补适配层，不需要反向污染 MClaw 内部模块边界。

## 入口和示例

- `setup.py` 已注册 `mclaw-train=mclaw.trainer.main:main`
- `examples/textcraft_train.sh` 展示了目标 CLI 形状

但要注意：当前 CLI 入口还没有真正实现，所以示例脚本仍是“未来接线方式”的说明，不是可直接运行的完整训练脚本。
