# MClaw: Monte-Carlo Tree Rollout for Agent RL

## 核心思想

Agent RL 中环境交互昂贵，LLM 推理相对便宜。MClaw 在每步决策时生成大量候选动作，通过聚类剪枝减少环境交互，用 Q-critic 为未执行动作估计 value，最终用 PPO 做 step-level policy 更新。

与 GRPO 对比：
- GRPO：每个 prompt 生成 n 条独立完整轨迹，outcome reward，训练信号 = n 个标量
- MClaw：第 0 轮生成 256 个候选动作，聚类选 16 个代表执行；后续每个活跃分支每步生成 16 个候选，聚类选 1 个执行。训练信号 = 16 条轨迹 × step-level advantage + 同簇未执行动作的 auxiliary loss

## 术语约定

| 术语 | 含义 |
|------|------|
| 候选动作 (candidate action) | 模型生成、尚未送入环境执行的动作 |
| 被执行动作 (executed action) | 最终送入环境执行的动作 |
| 真实分支 (executed branch) | 由被执行动作串起来的环境交互路径 |
| 同簇未执行动作 (sibling action) | 与被执行动作同簇但未执行的候选动作 |
| 本地批结构 (local batch) | `TrajectoryRecord`、`ActorBatch`、`AuxiliaryBatch`、`CriticBatch` |
| 协议 (protocol) | 对外部对象（tokenizer、推理引擎、环境、训练后端）要求的最小方法集合 |
| 适配层 (adapter layer) | 本地批结构 → 外部训练后端格式的桥接层，与 AgentGym-RL 接口语义兼容 |

## 算法流程

```
对每个 training step 的每个 prompt:

1. 初始化：创建 M=16 个环境实例，全部 reset 到同一个 item_id，获得 s_0

2. 第 0 轮（根节点展开）：
   a. vLLM 从 s_0 生成 K0=256 个候选动作
   b. FSDP forward: 256 个 [state+action] → 256 个 hidden states + 256 个 Q values
      （一次 forward 同时服务于聚类和 Q-critic）
   c. 用 hidden states 聚类为 16 簇，每簇选 1 个代表
   d. 16 个代表动作分配到 16 个环境实例执行 → (s'_j, r_j, done_j)
   e. 计算 advantage:
      - 被执行: Q = r_j + γ·V(s'_j)（真实 TD target）
      - 未执行: Q = Q-head 估计
      - V(s_0) = mean of 256 个 Q values
      - advantage = Q - V(s_0)
      - 同簇未执行动作保留为辅助样本

3. 后续每步 t = 1, 2, ..., max_rounds:
   a. 每个活跃分支生成 B=16 个候选动作
   b. 所有活跃分支的候选合并做一次 FSDP forward → hidden states + Q values
   c. 每个分支内聚类（C 个簇），选 1 个动作执行
   d. 执行 → (s'_j, r_j, done_j)，done 的分支终止
   e. advantage = (r_j + γ·V(s'_j)) - mean(分支内 16 个 Q values)
      同簇未执行动作保留为辅助样本

4. 构造训练数据:
   - 16 条完整轨迹: input_ids = [prompt | obs | action | obs | action | ...]
   - response_mask: 只在 assistant action tokens 上为 1
   - advantages: 每个 turn 填入该 step 的 advantage（标量广播到所有 token）

5. PPO actor update（经适配层对接外部训练后端）

6. Auxiliary loss: 同簇未执行动作共享被执行动作的 advantage
   L_aux = (1/C_sel) · Σ_c [(1/|S_c|) · Σ_{a∈S_c} l_aux(a)]（簇等权）

7. Q-head TD update（backbone frozen，只更新 Q-head）:
   Loss = MSE(QHead(h_action), r + γ·V(s'))
```

## 项目结构

```
MClaw/
├── plan.md
├── pyproject.toml
├── mclaw/
│   ├── config/
│   │   └── mclaw_trainer.yaml
│   ├── core/
│   │   ├── contracts.py             # 本地批结构和外部依赖协议
│   │   ├── tree_rollout.py          # 树状 rollout 引擎
│   │   ├── tree_node.py             # 树节点数据结构
│   │   └── branch_selector.py       # 聚类后的动作选择
│   ├── clustering/
│   │   ├── base.py                  # 聚类基类（含 K-Means + PCA 降维）
│   │   ├── hidden_state.py          # 主方案：last-layer hidden state
│   │   ├── output_grad.py           # Fallback：embedding(a) - E[embedding(·)]
│   │   ├── logprob.py               # Baseline：logprob 向量
│   │   └── logit_distribution.py    # Top-k logit 分布
│   ├── critic/
│   │   ├── q_head.py                # 2 层 MLP → 标量 Q value
│   │   ├── q_critic.py              # 共享 backbone + Q-head
│   │   └── advantage.py             # 树结构上的 step-level advantage
│   ├── trainer/
│   │   ├── contracts.py             # 训练后端协议与适配层
│   │   ├── mclaw_trainer.py         # 主训练循环
│   │   └── main.py                  # 入口点
│   └── utils/
│       └── vllm_hooks.py            # fallback 聚类用的 EmbeddingMatrixCache
├── examples/
│   └── textcraft_train.sh
```

## 模块设计

### TreeNode (`core/tree_node.py`)

```python
@dataclass
class TreeNode:
    state_tokens: List[int]
    action_tokens: List[int]
    parent: Optional['TreeNode']
    children: List['TreeNode']
    depth: int
    # 环境信息（仅被执行节点）
    executed: bool = False
    env_reward: float = 0.0
    env_next_state: Optional[str] = None
    done: bool = False
    # 模型信息（所有节点）
    log_prob: float = 0.0
    q_value: float = 0.0
    advantage: float = 0.0
    # 聚类信息
    cluster_id: int = -1
    cluster_feature: Optional[torch.Tensor] = None
    selected_for_aux_loss: bool = False
```

### 聚类 (`clustering/`)

基类接口：`extract_features(action_ids, state_ids, model_outputs) → (K, feat_dim)` + `cluster(features, n_clusters) → (labels, representatives)`

四种实现：

| | hidden_state | output_grad | logprob | logit_distribution |
|---|---|---|---|---|
| 语义质量 | 最高 | 中高 | 低 | 中低 |
| 额外计算 | 零（共享 Q-critic forward） | 矩阵乘法 ~2ms | 零 | 零 |
| 特征维度 | 3584（PCA→128） | 3584（PCA→128） | 变长 | k=100 |
| 与 Q-critic 对齐 | 完全对齐 | 间接 | 无 | 无 |
| 定位 | **主方案** | **fallback** | baseline | 不推荐单独用 |

### Q-Critic (`critic/`)

```python
class QHead(nn.Module):
    # 2层 MLP: hidden_dim → 1024 → 1024 → 1
    def forward(self, hidden_states: Tensor) -> Tensor:  # (batch, hidden_dim) → (batch,)

class QCritic:
    # backbone = rollout 阶段仍在 GPU 上的 FSDP module（零额外显存）
    # 推理: actor_module_fsdp(input_ids, output_hidden_states=True) → hidden → QHead → Q
    # 训练: backbone frozen, 只更新 QHead, Loss = MSE(Q, td_target)
```

### Advantage (`critic/advantage.py`)

```python
def compute_tree_advantage(tree_nodes, gamma=0.99):
    # 被执行节点: Q = r + γ·V(s')
    # 未执行节点: Q = Q-head 估计
    # V(s) = 同状态下所有候选动作 Q 值均值
    # advantage = Q(s,a) - V(s)
    # 同簇未执行动作继承 advantage，用于 auxiliary loss（簇等权汇总）
```

### TreeRollout (`core/tree_rollout.py`)

核心模块，语义对应 AgentGym-RL 的 `vllm_rollout.py:generate_sequences()`。

输入：batch of prompts
输出：`ActorBatch`（16 条轨迹 + step-level advantage）、`AuxiliaryBatch`（同簇未执行动作）、`CriticBatch`（TD 更新数据）

### MClawTrainer (`trainer/mclaw_trainer.py`)

```
每个 training step:
1. prompts = dataloader.next()
2. actor_data, aux_data, critic_data = tree_rollout.generate(prompts)
3. actor_backend.compute_log_prob(actor_data)
4. ref_policy.compute_ref_log_prob(actor_data)
5. actor_backend.update_policy(actor_data)       # PPO update
6. actor_backend.update_aux_loss(aux_data)        # 同簇 auxiliary loss
7. q_head.update(critic_data)                     # TD update
```

## 配置 (`mclaw_trainer.yaml`)

```yaml
mclaw:
  tree_rollout:
    root_budget: 256
    n_envs: 16
    root_clusters: 16
    branch_budget: 16
    intra_branch_clusters: 4
    max_rounds: 30
  clustering:
    method: hidden_state          # hidden_state | output_grad | logprob | logit_distribution
    pca_dim: 128
  q_critic:
    hidden_dim: 3584
    intermediate_dim: 1024
    lr: 1e-4
    gamma: 0.99
  aux_loss:
    coef: 0.2
    weighting: equal_per_selected_cluster

algorithm:
  adv_estimator: mclaw
```

## 实现顺序

**Phase 1 — 基础框架 ✅:** 项目骨架 + `tree_node.py` + `q_head.py` + `clustering/base.py`

**Phase 2 — 核心引擎 ✅:** `logprob.py`（baseline 聚类）→ `q_critic.py` → `advantage.py` → `branch_selector.py` → `tree_rollout.py`

**Phase 3 — 训练循环 ✅:** `mclaw_trainer.py` + `main.py` + 配置文件 + 示例脚本

**Phase 4 — 聚类升级 ✅:** `hidden_state.py`（主方案）→ `output_grad.py`（fallback）→ `vllm_hooks.py`

**Phase 5 — 集成（进行中）:** 适配层对接 AgentGym-RL ✅ → agentenv 环境集成 ✅ → 端到端联调 🔜

## 关键设计决策

### 1. Q-critic backbone → 复用 rollout 阶段的 FSDP module

在 `with rollout_sharding_manager:` 上下文内，FSDP 权重仍在 GPU 上（AgentGym-RL 的 `fsdp_vllm.py` 未做 offload），可直接用于 PyTorch forward。零额外显存，零权重同步。缺点是无 KV cache 复用，但对昂贵环境（WebArena ~10s/step）不是瓶颈。后续可迁移到 vLLM hidden state hook 优化。

### 2. 环境实例管理 → 固定 16 个实例，不做 fork

每个 prompt 预分配 16 个环境实例，reset 到同一初始状态。第 0 轮聚类选出 16 个代表各执行到一个实例，之后每个实例沿自己的分支继续。不需要环境原生 fork 支持。done 的分支终止，实例空闲不复用。

后续扩展（如需动态分支分配）：TextCraft/BabyAI 用 deepcopy fork；WebArena 用 replay；SearchQA 用文本复制。

### 3. 训练数据 → 完整轨迹 + step-level advantage

16 条完整轨迹作 PPO 主训练数据，`response_mask` 过滤 observation tokens，每个 turn 的 advantage 广播到 assistant tokens。同簇未执行动作进入 auxiliary loss（簇等权）。其他未执行动作的价值体现在 V(s) 估计中。

### 4. 聚类方案 → hidden_state 主方案

FSDP forward 同时服务 Q-critic 和聚类，零额外计算。Hidden state 是模型对 (state, action) 的完整内部表示，且与 Q-head 输入完全对齐——聚类的"相似"直接意味着"Q value 相近"，确保选出的代表在 value 空间上多样化。
