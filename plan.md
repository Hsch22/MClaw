# MClaw: Monte-Carlo Tree Rollout for Agent RL

## 核心思想

在 agent RL 中，环境交互昂贵，LLM 推理相对便宜。MClaw 在每一步 action 决策时，用大量候选动作（candidate actions）做探索，通过聚类剪枝减少环境交互次数，用 Q-critic 为未执行动作估计 value，最终用 PPO 做 step-level policy 更新。

与传统 GRPO 的对比：
- GRPO：每个 prompt 生成 n 条独立完整轨迹，outcome reward，训练信号 = n 个标量
- MClaw：第 0 轮从根状态一次生成 K0=256 个候选动作，全局聚类后选 M=16 个代表执行，形成 16 条真实分支；后续每条活跃分支每步固定生成 B=16 个候选动作，只做分支内聚类并选 1 个被执行动作。训练信号 = 最多 16 条完整轨迹 × 每步 step-level advantage + 被选簇内未执行同簇动作的 auxiliary loss

## 术语约定

- 候选动作（candidate action）：模型在某一步生成、但尚未送入真实环境执行的动作。
- 被执行动作（executed action）：同一步候选动作中最终真正送入环境执行的动作。
- 真实分支（executed branch）：由被执行动作串起来的真实环境交互路径。
- 同簇未执行动作（sibling action）：与被执行动作属于同一聚类簇、但未执行的候选动作。
- 辅助样本（auxiliary sample）：由同簇未执行动作构造出的 auxiliary loss 输入。
- 本地批结构（local batch）：MClaw 内部统一使用的 `TrajectoryRecord`、`ActorBatch`、`AuxiliaryBatch`、`CriticBatch`。
- 协议（protocol）：MClaw 对 tokenizer、推理引擎、环境客户端、训练后端等外部对象要求的最小方法集合。
- 适配层（adapter layer）：把本地批结构转换为外部训练后端可接受格式的桥接层。MClaw 与 AgentGym-RL 目标是接口语义兼容，但不直接 `import` 其源码。

## 算法流程

```
对每个 training step 的每个 prompt:

1. 初始化：
   - 创建 M=16 个环境实例，全部 reset 到同一个 item_id
   - 获得初始状态 s_0（所有实例共享）
   - active_branches = [s_0]（初始只有 1 个根状态，还没有真实分支）

2. 第 0 轮（根节点展开）：

   a. 生成根节点候选动作:
      - 从根状态 s_0 一次生成 K0 = 256 个候选动作
      - [vLLM batch forward]

   b. FSDP backbone forward（聚类 + Q 估计共享，见设计决策 §1 §4）:
      - 对所有 256 个 [state + action] 做一次 FSDP batch forward
      - 获得 256 个 hidden states（用于聚类）和 256 个 Q values（用于 advantage）
      - 这一次 forward 同时服务于聚类和 Q-critic，零额外计算

   c. 根节点全局聚类选代表:
      - 用 hidden states 做聚类（主方案）或 output_grad 特征（fallback）
      - 聚类为 M=16 个簇，每个簇选一个代表动作
      - 聚类确保代表动作在语义上多样化

   d. 执行代表动作:
      - 16 个代表动作分配到 16 个环境实例上执行
      - 每个环境实例执行一个 action → 获得 (s'_j, r_j, done_j)
      - 执行后形成 16 条真实分支；后续只维护这 16 个环境实例，不再新增实例，也不做 fork

   e. 计算 step-level advantage（用步骤 b 已有的 Q values）:
      - 被执行的 16 个: Q = r_j + γ * V(s'_j)  [真实 TD target，覆盖 Q-head 估计]
      - 未执行的 240 个: Q = Q-head(h_s, h_a)  [步骤 b 已计算]
      - V(s_0) = mean of all 256 个候选动作的 Q values
      - 对被执行动作: advantage = Q_td - V(s_j)
      - 将 advantage 记录到对应分支的轨迹上
      - 每个被执行动作所在簇的其余候选动作不执行环境，但保留为辅助样本参与 loss 计算

3. 后续每一步 t = 1, 2, ..., max_rounds:

   a. 对每个活跃分支 s_j:
      - 固定生成 B = 16 个下一步候选动作
      - 如果当前活跃分支数为 A，则该步总候选动作数 = 16 × A

   b. FSDP backbone forward（批量合并所有活跃分支的候选动作）:
      - 对所有 [state_j + action_{j,k}] 做一次 batch forward
      - 获得所有 hidden states 和 Q values

   c. 分支内聚类 + 选执行动作:
      - 对每个分支自己的 16 个候选动作做分支内聚类
      - 聚成 C 个簇（C 为可配置超参）
      - 从分支内候选中最终只选 1 个 action 真实执行
      - 该动作所在簇的其余候选动作不执行环境，但保留为辅助样本

   d. 执行：
      - 每个活跃分支对应的环境实例最多只执行 1 个 action
      - 获得 (s'_j, r_j, done_j)
      - done 的分支终止；实例不复用到新 fork

   e. 计算 step-level advantage:
      - V(s_j) = mean of Q values of 该分支的 16 个候选动作
      - 对被执行动作: Q_td = r_j + γ * V(s'_j)
      - advantage_j = Q_td - V(s_j)
      - 将 advantage 记录到该分支当前 step 的 assistant tokens
      - 被执行动作所在簇的同簇未执行动作共享该 step 的训练信号，用于 auxiliary loss
      - auxiliary loss 采用“簇等权”汇总：先对每个被选簇内部求均值，再对所有被选簇求平均，避免大簇吞掉小簇

   f. 如果所有分支 done，结束

4. 构造训练数据（完整轨迹格式，见设计决策 §3）:
   - 最多 16 条被执行的完整轨迹，每条包含多轮 [user_obs | assistant_action | ...]
   - response_mask: 只在 assistant action tokens 上为 1
   - advantages: 每个 turn 的 assistant tokens 填入该 step 的 advantage
   - MClaw 内部先组织为 `TrajectoryRecord` / `ActorBatch`，再由适配层转换为外部训练后端需要的格式

5. PPO 更新（actor only）:
   - 目标语义对齐外部 PPO actor 更新接口（如 AgentGym-RL 的 `dp_actor.py:update_policy()`）
   - 具体接入通过训练后端协议和适配层完成，不直接 `import` 外部源码
   - compute_policy_loss 逐 token 计算，response_mask 自动过滤 env observation tokens

6. 同簇 auxiliary loss:
   - 对每个被执行动作，取其所在簇的其余候选动作
   - 这些同簇未执行动作不执行环境，但共享该 step 的 advantage / TD target 作为软监督
   - 权重按“簇等权”处理：每个被选簇先做簇内平均，不按样本数直接累加
   - 若某步有 C_sel 个被选簇参与 auxiliary loss，第 c 个簇有 |S_c| 个同簇未执行动作，则
     L_aux = (1 / C_sel) * Σ_c [ (1 / |S_c|) * Σ_{a in S_c} l_aux(a) ]
   - prototype 默认作为 auxiliary 项叠加到主 loss 上，整体系数由超参控制

7. Q-head 更新（Q-head only，backbone frozen）:
   - 用被执行的 (s, a, r, s') 做 TD 更新
   - Loss: MSE(QHead(backbone([s+a])[-1]), td_target)
```

## 项目结构

```
MClaw/
├── README.md
├── plan.md                          # 本文件
├── setup.py
├── mclaw/
│   ├── __init__.py
│   ├── README.md
│   ├── config/
│   │   ├── __init__.py
│   │   ├── README.md
│   │   └── mclaw_trainer.yaml       # 默认训练配置骨架
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── README.md
│   │   ├── contracts.py             # 本地批结构和外部依赖协议
│   │   ├── tree_rollout.py          # 树状 rollout 引擎（替代 vllm_rollout.py 的 generate_sequences）
│   │   ├── tree_node.py             # 树节点数据结构
│   │   └── branch_selector.py       # 根节点代表选择 + 分支内单动作选择
│   │
│   ├── clustering/
│   │   ├── __init__.py
│   │   ├── README.md
│   │   ├── base.py                  # 聚类基类接口
│   │   ├── output_grad.py           # Output-layer 梯度近似: embedding(a) - E_p[embedding(·)]
│   │   ├── hidden_state.py          # Hidden state 聚类
│   │   ├── logprob.py               # Logprob 向量聚类（baseline）
│   │   └── logit_distribution.py    # Top-k logit distribution 聚类
│   │
│   ├── critic/
│   │   ├── __init__.py
│   │   ├── README.md
│   │   ├── q_head.py                # Q-head 模块定义（轻量 MLP）
│   │   ├── q_critic.py              # Q-critic：共享 backbone + Q-head，backbone frozen
│   │   └── advantage.py             # step-level advantage 计算（树结构上的 TD）
│   │
│   ├── trainer/
│   │   ├── __init__.py
│   │   ├── README.md
│   │   ├── contracts.py             # 训练后端协议与适配边界
│   │   ├── mclaw_trainer.py         # 主训练循环（替代 ray_trainer.py 的 fit()）
│   │   └── main.py                  # 入口点
│   │
│   └── utils/
│       ├── __init__.py
│       ├── README.md
│       └── vllm_hooks.py            # vLLM 扩展：返回 hidden states / logits
│
├── examples/
│   ├── README.md
│   └── textcraft_train.sh           # 示例训练脚本
```

## 模块详细设计

### 1. `mclaw/core/tree_node.py` — 树节点

```python
@dataclass
class TreeNode:
    state_tokens: List[int]          # 当前状态的 token 序列（完整 context）
    action_tokens: List[int]         # 该节点的 action token 序列（为空则是根节点）
    parent: Optional['TreeNode']
    children: List['TreeNode']
    depth: int

    # 环境信息（仅被执行的节点有）
    executed: bool = False
    env_reward: float = 0.0          # step reward from environment
    env_next_state: Optional[str] = None  # environment observation
    done: bool = False

    # 模型信息（所有节点都有）
    log_prob: float = 0.0            # log π(a|s)
    q_value: float = 0.0            # Q(s, a) — 真实 TD target 或 critic 估计
    advantage: float = 0.0

    # 聚类信息
    cluster_id: int = -1
    cluster_feature: Optional[torch.Tensor] = None  # 聚类用的特征向量
    selected_for_aux_loss: bool = False
```

### 2. `mclaw/clustering/base.py` — 聚类接口

```python
class BaseClusterer(ABC):
    """所有聚类方案的基类"""

    @abstractmethod
    def extract_features(
        self,
        action_token_ids: List[List[int]],   # K 个 action 的 token ids
        state_token_ids: List[int],           # 共享的 state token ids
        model_outputs: Dict[str, Any],        # 额外信息（hidden_states, logprobs, embedding_matrix 等）
    ) -> torch.Tensor:
        """提取聚类特征，返回 (K, feature_dim)"""
        pass

    def cluster(
        self,
        features: torch.Tensor,  # (K, feature_dim)
        n_clusters: int,         # 目标簇数 M
    ) -> Tuple[List[int], List[int]]:
        """
        返回:
          cluster_labels: 每个 action 的簇 id，长度 K
          representative_ids: 每个簇的代表 action index，长度 M
        默认实现用 K-Means，子类可以 override。
        高维特征（hidden_state, output_grad）先 PCA 降维到 64-128 维再聚类。
        """
        pass
```

四种聚类实现（按推荐优先级排序）：

#### `hidden_state.py` — Hidden state 聚类（主方案，推荐）
```
特征 = 生成 action 最后一个 token 时的 last-layer hidden state
维度: hidden_dim (3584 for Qwen2.5-7B)

来源: FSDP backbone forward 的 output.hidden_states[-1]
      这次 forward 和 Q-critic 估计共享（见设计决策 §4），零额外计算。

语义质量: 最高。hidden state 是模型对 (state, action) 对的完整内部表示，
         编码了 action 的语义内容、与 state 的关系、以及模型对后续的预期。
         两个 hidden state 相似的 action，在模型看来确实是"做了类似的事情"。

和 Q-critic 的对齐: Q-head 的输入就是 hidden state。聚类的"相似"直接意味着
                   "Q value 相近"，这正是我们想要的 — 选出 Q value 多样化的代表。

注意事项:
- 根节点 256 个样本、分支内 16 个样本都属于"样本数小于维度"的场景
- 需要 PCA 降维到 64-128 维再聚类
- 聚类发生在 FSDP forward 之后（不是 vLLM generate 之后），
  这改变了流程顺序：生成 → FSDP forward → 聚类 → 执行 → advantage
```

#### `output_grad.py` — Output-layer 梯度近似（fallback 方案）
```
特征 = embedding(a_t) - E_{p(·|s,a_{<t})}[embedding(·)]
     = embedding(a_t) - softmax(logits) @ embedding_matrix
维度: hidden_dim (3584 for Qwen2.5-7B)

物理意义: 实际采样 token 的 embedding 与 policy 期望 embedding 的偏差。
         度量的是"这个 action 相对于 policy 的平均行为偏离了多少、往哪个方向偏离"。
         数学上是 cross-entropy loss 对 last hidden state 梯度的近似。

需要: action token ids + top-k logprobs（vLLM 返回）+ embedding matrix（从 actor model 取）
成本: 一次矩阵乘法 (K, top_k) @ (top_k, hidden_dim) ≈ 1-2ms on GPU

优势: 不需要额外 forward pass，只用 vLLM 的标准输出 + embedding matrix。
     适用于不想做 FSDP forward 的场景（比如 Q-critic 被禁用时的纯聚类模式）。

注意事项:
- 用 top-k logprobs 近似完整 softmax 分布（k=200 时近似误差可接受）
- embedding matrix 在 FSDP 下是 sharded 的，需要在 rollout 前 all-gather 一次缓存
- use_mean_pooling=true 时对 action 所有 token 位置取均值，否则只看第一个 token
- 只看第一个 token 时，以相同动词开头的不同 action（如 "click [buy]" vs "click [cancel]"）
  可能被误判为相似
```

#### `logprob.py` — Logprob 向量聚类（baseline）
```
特征 = action 中每个 token 的 log_prob 拼接成向量
维度: action_length（变长，需 padding）

需要: vLLM 默认返回的 logprobs（SamplingParams(logprobs=1)，当前代码已设置）
成本: 零

语义质量: 低。只反映 policy 对每个 token 的置信度模式，不反映 action 的语义内容。
         两个语义完全不同的 action 如果置信度模式相似会被聚到一起。

用途: 验证聚类 → 执行 → Q 估计的端到端流程正确性。
     实现最简单，适合 Phase 1 的 baseline。

注意事项:
- 变长 padding 导致短 action 的特征被大量零稀释，聚类质量差
- 不同 step 的 action 长度分布可能差异大，聚类超参数难以统一
```

#### `logit_distribution.py` — Top-k logit distribution 聚类（不推荐单独使用）
```
特征 = action 第一个 token 位置的 top-k logit 值
维度: k（如 100）

需要: vLLM 返回 top-k logprobs（调大 SamplingParams.logprobs）
成本: 零

语义质量: 中低。捕捉了"决策分支点"的信息 — action 第一个 token 往往决定 action 类型
         （如 "click" vs "type" vs "scroll"），top-k 分布能区分这些大类。
         但完全忽略后续 token，同类型不同目标的 action 无法区分。

注意事项:
- 只看第一个 token，丢失 action 后续内容的信息
- top-k 向量是稀疏的，两个 action 的 top-k 集合可能不重叠，
  欧氏距离效果差，需要 KL/JS 散度（和标准 K-Means 不兼容）
- 可作为粗筛层：先按 action 类型分大类，再在类内用更精细的方法
```

### 3. `mclaw/critic/q_head.py` — Q-head 模块

```python
class QHead(nn.Module):
    """
    轻量级 Q-head，接在 frozen backbone 之后。

    架构: 共享 state encoding
    - backbone 处理 [state + action] 序列
    - 取 action 最后一个 token 的 hidden state
    - 过 2 层 MLP → 标量 Q value

    训练时只更新 QHead 参数，backbone 完全 frozen。
    """
    def __init__(self, hidden_dim: int, intermediate_dim: int = 1024):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch_size, hidden_dim) — action 最后一个 token 的 hidden state
        Returns:
            q_values: (batch_size,)
        """
        return self.mlp(hidden_states).squeeze(-1)
```

### 4. `mclaw/critic/q_critic.py` — Q-Critic

```python
class QCritic:
    """
    Q(s, a) = QHead(backbone([state_tokens + action_tokens])[-1])

    backbone 来源（选项 C）: 直接复用 rollout 阶段仍在 GPU 上的 FSDP module。

    原理: 在 AgentGym-RL 中，FSDPVLLMShardingManager.__enter__() 将 FSDP 权重
    同步到 vLLM 后，FSDP module 的权重并未被 offload 到 CPU（见 fsdp_vllm.py:94-98
    的 TODO 注释）。因此在 `with rollout_sharding_manager:` 上下文内，FSDP module
    仍然可用于标准 PyTorch forward。

    推理流程（根节点 256 个 candidate；后续每个活跃分支 16 个 candidate）:
    1. 构造 batch: 将 [state_tokens + action_tokens] 拼接为 input_ids
       - 同一 state 下的多个 action 共享 state prefix
       - batch forward 一次处理所有 candidate
    2. FSDP backbone forward:
       output = actor_module_fsdp(input_ids, output_hidden_states=True)
       hidden = output.hidden_states[-1]  # (batch, seq_len, hidden_dim)
    3. 取每个 action 最后一个 token 的 hidden state:
       h_action = hidden[i, action_last_token_pos, :]  # (hidden_dim,)
    4. QHead forward: q_value = QHead(h_action)

    训练流程:
    - 只用被执行的 (s, a, r, s') 数据
    - TD target: y = r + γ * V(s')，其中 V(s') = mean Q of s' 的 children
    - Loss: MSE(QHead(backbone([s+a])[-1]), y)
    - 只更新 QHead 参数，backbone frozen (requires_grad=False)
    """
```

关键设计决策：
- **选项 C: 复用 rollout 阶段的 FSDP module 做 Q-critic forward。** 在 `with rollout_sharding_manager:` 上下文内，FSDP 权重仍在 GPU 上（未被 offload），可以直接用 `actor_module_fsdp(input_ids, output_hidden_states=True)` 做标准 PyTorch forward 拿 hidden state。零额外显存，零权重同步，标准 PyTorch 接口。
- QHead 参数量很小（~几 MB），不需要 FSDP，直接放 GPU。
- 训练时 backbone frozen（`requires_grad=False`），只 backward 到 QHead。
- 后续优化路径：如果 Q 估计成为性能瓶颈（根节点 256 个 candidate / 活跃分支总数较多时的 FSDP forward 没有 KV cache 复用），可以迁移到选项 A（vLLM hidden state hook）获得 KV cache 复用。但 prototype 阶段选项 C 足够。

### 5. `mclaw/critic/advantage.py` — Step-level Advantage

```python
def compute_tree_advantage(
    tree_nodes: List[TreeNode],  # 一棵树的所有节点
    gamma: float = 0.99,
) -> None:
    """
    在树结构上计算 step-level advantage，原地更新每个节点的 advantage 字段。

    对于被执行的节点（有真实 env reward）:
        Q(s, a) = r + γ * V(s')
        其中 V(s') = mean of Q values of s' 的 children（如果有），否则用 Q-head 估计

    对于未执行的节点:
        Q(s, a) = Q-head 估计

    advantage = Q(s, a) - V(s)
    其中 V(s) = 同一状态下所有候选动作的 Q(s, a_i) 均值

    对于被执行动作所在簇的同簇未执行动作:
        不执行环境，但继承该 step 的 advantage / TD target
        作为 auxiliary loss 的软监督项
        汇总方式采用簇等权：先簇内平均，再簇间平均
    """
```

### 6. `mclaw/core/tree_rollout.py` — 树状 Rollout 引擎

这是核心模块，目标语义上对应 AgentGym-RL 的 `vllm_rollout.py:generate_sequences()`，但 MClaw 内部会先生成自己的本地批结构，再通过适配层对接外部训练后端。

```python
class TreeRollout:
    """
    树状 rollout 引擎。

    与 AgentGym-RL 的 vLLMRollout 的关系:
    - 复用 vLLM inference engine 做 action 生成
    - 复用 env_client 做环境交互
    - 替换扁平的 while 循环为树状展开
    - 新增: 聚类、Q-critic 估计（FSDP backbone forward）、分支管理

    输入: 一个 batch 的 prompts（语义上对齐外部训练数据加载器输出）
    输出:
      - actor_data: `ActorBatch`，包含最多 M=16 条完整轨迹
      - aux_actor_data: `AuxiliaryBatch`，包含同簇未执行动作对应的辅助样本
      - critic_data: `CriticBatch`，包含被执行的 (s, a, r, s') 用于 Q-head TD 更新
    """

    def __init__(
        self,
        inference_engine,          # vLLM LLM instance
        actor_module_fsdp,         # FSDP wrapped backbone（选项 C，用于 Q-critic forward）
        q_critic: QCritic,         # Q-critic 接口，内部持有 Q-head
        clusterer: BaseClusterer,
        tokenizer,
        config: TreeRolloutConfig,
        env_client_factory=None,
        branch_selector=None,
    ):
        pass

    def generate_tree_rollout(
        self,
        prompts,
    ) -> TreeRolloutOutput:
        """
        主循环（伪代码）:

        核心设计:
        - 第 0 轮: 256 个根节点候选动作，全局聚类选 16 个代表
        - 后续轮: 每个活跃分支固定 16 个候选动作，只做分支内聚类
        - FSDP forward 同时服务于聚类和 Q 估计（见设计决策 §4）

        for each prompt in batch:
            # 1. 初始化 16 个环境实例
            env_pool = [init_env_client() for _ in range(M)]
            for client in env_pool:
                client.reset(item_id)
            s_0 = env_pool[0].observe()

            # 每个环境实例对应一个轨迹处理器
            # 实现上只要求兼容 RolloutHandlerProtocol，不要求直接复用外部类
            handlers = [trajectory_handler_factory() for _ in range(M)]

            # 2. 第 0 轮：根状态一次生成 256 个候选动作
            root_prompt_ids = build_prompt_from_state(s_0)
            root_outputs = inference_engine.generate(
                prompt_token_ids=[root_prompt_ids] * root_budget, ...
            )
            root_actions = extract_actions(root_outputs)
            root_state_tokens = tokenize_state(s_0)

            # 3. 根节点 FSDP forward
            with torch.no_grad():
                q_input_ids = [root_state_tokens + action_tokens for action_tokens in root_actions]
                output = actor_module_fsdp(q_input_ids, output_hidden_states=True)
                hidden = output.hidden_states[-1]
                h_actions = hidden[:, action_last_pos, :]
                q_values = q_head(h_actions)
                root_v = q_values.mean()

            # 4. 根节点全局聚类，选 16 个代表执行
            root_cluster_labels, representative_ids = clusterer.cluster(
                h_actions, n_clusters=M
            )
            for j, rep_id in enumerate(representative_ids):
                rep_action = root_actions[rep_id]
                handlers[j].add_assistant_message(tokenizer, rep_action)
                step_output = env_pool[j].step(rep_action)
                handlers[j].score = step_output.reward
                handlers[j].done = step_output.done
                root_executed_q = step_output.reward + gamma * estimate_next_v(step_output)
                handlers[j].step_advantages.append(root_executed_q - root_v)
                mark_auxiliary_samples_for_loss(
                    root_cluster_labels,
                    rep_id,
                    root_executed_q - root_v,
                    weight_mode="equal_per_selected_cluster",
                )
                if not step_output.done:
                    handlers[j].add_user_message(tokenizer, step_output.state)

            # 5. 后续轮：每个活跃分支固定展开 16 个候选动作
            for step in range(1, max_rounds):
                active_branches = [(j, handlers[j]) for j in range(M) if not handlers[j].done]
                if len(active_branches) == 0:
                    break

                all_prompts = []
                branch_offsets = {}
                for j, handler in active_branches:
                    prompt_ids = handler.get_generation_prompt(tokenizer)
                    branch_offsets[j] = len(all_prompts)
                    all_prompts.extend([prompt_ids] * branch_budget)
                outputs = inference_engine.generate(prompt_token_ids=all_prompts, ...)

                # 6. 所有活跃分支的候选动作合并做一次 FSDP forward
                with torch.no_grad():
                    q_input_ids = build_state_action_batch(active_branches, all_actions)
                    output = actor_module_fsdp(q_input_ids, output_hidden_states=True)
                    hidden = output.hidden_states[-1]
                    h_actions = hidden[:, action_last_pos, :]
                    q_values = q_head(h_actions)

                # 7. 每个分支单独聚类，从本分支 16 个候选动作中只选 1 个执行
                for j, handler in active_branches:
                    start = branch_offsets[j]
                    end = start + branch_budget
                    branch_features = h_actions[start:end]
                    branch_q = q_values[start:end]
                    branch_labels, branch_rep_ids = clusterer.cluster(
                        branch_features, n_clusters=intra_branch_clusters
                    )
                    chosen_id = select_one_action(branch_rep_ids, branch_q)
                    chosen_action = all_actions[start + chosen_id]

                    handlers[j].add_assistant_message(tokenizer, chosen_action)
                    step_output = env_pool[j].step(chosen_action)
                    handlers[j].score = step_output.reward
                    handlers[j].done = step_output.done

                    v_s = branch_q.mean()
                    executed_q = step_output.reward + gamma * estimate_next_v(step_output)
                    handlers[j].step_advantages.append(executed_q - v_s)
                    mark_auxiliary_samples_for_loss(
                        branch_labels,
                        chosen_id,
                        executed_q - v_s,
                        weight_mode="equal_per_selected_cluster",
                    )
                    if not step_output.done:
                        handlers[j].add_user_message(tokenizer, step_output.state)

            # 8. 构造训练数据（16 条完整轨迹 + 辅助样本）
            for j in range(M):
                # handlers[j] 已经包含完整的多轮交互历史
                # 构造 response_mask（只在 assistant tokens 上为 1）
                # 构造 advantages（每个 turn 填入对应的 step advantage）
                # MClaw 内部先输出 TrajectoryRecord / ActorBatch

        return TreeRolloutOutput(...)
        """
```

### 7. `mclaw/core/branch_selector.py` — 候选选择

```python
class BranchSelector:
    """
    不再负责动态分配分支预算。

    当前职责:
    - 根节点: 从 256 个候选动作的 16 个簇中各选 1 个代表
    - 分支内部: 在某个分支自己的 16 个候选动作中，
      先做聚类，再从代表集合里选 1 个真实执行动作
    - 标记被执行动作所在簇的同簇未执行动作，供 auxiliary loss 使用
    - 生成簇等权的 sample weight，避免大簇在 auxiliary loss 中吞掉小簇
    """
```

### 8. `mclaw/trainer/mclaw_trainer.py` — 训练循环

```python
class MClawTrainer:
    """
    主训练循环，替代 AgentGym-RL 的 RayPPOTrainer.fit()。

    每个 training step:
    1. 从 dataloader 取一个 batch of prompts
    2. tree_rollout.generate_tree_rollout(prompts)
       → actor_data: `ActorBatch`（step-level advantage 已填入）
       → aux_actor_data: `AuxiliaryBatch`
       → critic_data: `CriticBatch`
    3. actor_backend.compute_log_prob(actor_data)  # recompute old_log_probs
    4. ref_policy.compute_ref_log_prob(actor_data)  # KL penalty
    5. actor_backend.update_policy(actor_data)  # PPO update，经适配层对接外部后端
    6. actor_backend.update_aux_loss(aux_actor_data)  # 同簇未执行动作的 auxiliary loss（簇等权）
    7. q_head.update(critic_data)  # TD update，只更新 Q-head 参数
    8. log metrics

    与 AgentGym-RL 的 RayPPOTrainer 的关系:
    - 对齐: Ray worker group、FSDP、checkpoint、logging 的调用阶段和数据语义
    - 对齐: 外部 PPO actor update 所需的完整轨迹格式 + step-level advantage 语义
    - 替换: rollout 阶段（扁平 → 树状）、advantage 计算（outcome → step-level）
    - 新增: Q-head 模块、聚类模块、环境实例池管理、本地批结构和适配层
    """
```

### 9. `mclaw/utils/vllm_hooks.py` — vLLM 扩展

```python
"""
扩展 vLLM 的输出，使其返回聚类所需的额外信息。

当前状态: 主方案（hidden_state 聚类）不需要 hack vLLM，因为 hidden states
通过 FSDP backbone forward 获取（和 Q-critic 共享同一次 forward）。

本模块仅在 fallback 方案（output_grad 聚类）中使用，用于提取 embedding matrix
和处理 top-k logprobs。

方案 1（output_grad 聚类 — fallback）:
  - 不需要改 vLLM 的 generate 流程
  - 需要: top-k logprobs（调大 SamplingParams.logprobs=200）+ embedding matrix
  - embedding matrix 从 FSDP module 取: actor_module_fsdp.get_input_embeddings().weight
  - 在 FSDP 下 embedding 是 sharded 的，需要在 rollout 前 all-gather 一次缓存

方案 2（hidden_state 聚类 — 后续优化，当前不需要）:
  - 如果后续迁移到选项 A（vLLM 做 Q-critic forward），需要 hook vLLM 返回 hidden states
  - 实现: 注册 forward hook 到 model 的最后一层，缓存 hidden states
  - 和 vLLM 版本强耦合，维护成本高
"""

class EmbeddingMatrixCache:
    def __init__(self, actor_module_fsdp):
        """
        在 rollout 开始前，从 FSDP module 中 all-gather embedding matrix 并缓存。
        embedding matrix 在整个 rollout 阶段不变（actor 权重在 rollout 期间 frozen）。
        """
        pass

    def get_embedding_matrix(self) -> torch.Tensor:
        """返回完整的 embedding matrix, shape (vocab_size, hidden_dim)"""
        pass
```

## 配置文件设计 (`mclaw_trainer.yaml`)

```yaml
# 配置语义对齐 AgentGym-RL，具体加载实现保留在 MClaw 内部
defaults:
  - _self_

mclaw:
  # 树状 rollout 参数
  tree_rollout:
    root_budget: 256           # 第 0 轮根状态 rollout 数
    n_envs: 16                 # 固定环境实例数 = 固定真实分支数 M
    root_clusters: 16          # 根节点全局聚类簇数
    branch_budget: 16          # 每个活跃分支每步的候选动作数
    intra_branch_clusters: 4   # 分支内聚类簇数 C
    max_rounds: 30             # 最大交互轮数

  # 聚类方案
  clustering:
    method: hidden_state        # hidden_state（主方案）| output_grad（fallback）| logprob（baseline）| logit_distribution
    pca_dim: 128               # 高维特征（hidden_state, output_grad）聚类前 PCA 降维的目标维度
    # hidden_state 特有参数
    hidden_state:
      layer: -1                # 取哪一层的 hidden state（-1 = 最后一层）
    # output_grad 特有参数（fallback）
    output_grad:
      use_mean_pooling: true   # action 多 token 时取均值 vs 只取第一个 token
      top_k_logprobs: 200      # 用 top-k logprobs 近似 softmax 分布
    # logit_distribution 特有参数
    logit_distribution:
      top_k: 100

  # Q-critic 参数
  q_critic:
    hidden_dim: 3584           # 和 backbone hidden_dim 一致
    intermediate_dim: 1024
    lr: 1e-4
    gamma: 0.99
    update_freq: 1             # 每个 training step 更新几次 Q-head

  aux_loss:
    coef: 0.2                          # 同簇未执行动作的 auxiliary loss 总权重
    use_same_advantage: true           # 同簇未执行动作共享被执行动作的 advantage
    weighting: equal_per_selected_cluster  # 先簇内平均，再簇间平均

# 以下字段语义上对齐外部训练框架
algorithm:
  adv_estimator: mclaw         # 新增的 advantage estimator 类型
  kl_ctrl: ...

actor_rollout_ref: ...
data: ...
trainer: ...
```

## 实现顺序

### Phase 1: 基础框架
1. 项目骨架 + 配置文件
2. `tree_node.py` — 树节点数据结构
3. `q_head.py` — Q-head 模块
4. `base.py` (clustering) — 聚类基类 + K-Means 默认实现（含 PCA 降维）

### Phase 2: 核心引擎（先跑通端到端流程）
5. `logprob.py` — 最简单的 baseline 聚类，用于验证流程
6. `q_critic.py` — Q-critic 实现（FSDP backbone forward + Q-head）
7. `advantage.py` — step-level advantage 计算
8. `branch_selector.py` — 根节点代表选择 + 分支内单动作选择
9. `tree_rollout.py` — 树状 rollout 引擎（用 logprob 聚类先跑通）

### Phase 3: 训练循环
10. `mclaw_trainer.py` — 主训练循环
11. `main.py` — 入口点
12. `mclaw_trainer.yaml` — 配置文件
13. `textcraft_train.sh` — 示例脚本

### Phase 4: 聚类方案升级
14. `hidden_state.py` — 主方案，复用 Q-critic 的 FSDP forward（零额外计算）
15. `output_grad.py` — fallback 方案 + `vllm_hooks.py`（EmbeddingMatrixCache）
16. `logit_distribution.py` — 可选

### Phase 5: 集成与调试
17. 通过适配层与 AgentGym-RL 风格的 Ray worker 架构集成
18. 与 agentenv 环境集成
19. 端到端测试

## 已确定的设计决策

### 1. Q-head 的 backbone 来源 → 选项 C：复用 rollout 阶段的 FSDP module

Q-head 复用 actor 的 backbone（同一个 FSDP module 实例）。

在 AgentGym-RL 中，`FSDPVLLMShardingManager.__enter__()` 将 FSDP 权重同步到 vLLM 后，FSDP module 的权重并未被 offload 到 CPU（见 `fsdp_vllm.py:94-98` 的 TODO 注释）。因此在 `with rollout_sharding_manager:` 上下文内，FSDP module 仍然可用。

Q-critic 直接用 `actor_module_fsdp(input_ids, output_hidden_states=True)` 做标准 PyTorch forward 拿 hidden state，再过 Q-head MLP。

优势：
- 零额外显存（不需要第二份 backbone 权重）
- 零权重同步（就是同一个 module）
- 标准 PyTorch 接口，`output.hidden_states[-1]` 直接拿 hidden state，无需 hack vLLM
- 调试简单，可以直接 print、断点、grad check

劣势：
- 没有 KV cache 复用。根节点 256 个 [state + action]、后续每步最多 16 × active_branches 个 batch forward 会重复计算 state prefix 的 attention。
- FSDP forward 会触发 all-gather 通信。

性能评估：对于 256 个 candidate 的 batch forward，假设平均序列长度 2000 tokens，在 8 卡 A100 上约需几秒到十秒量级。相比环境交互延迟（TextCraft ~100ms/step × 16 = 1.6s，WebArena ~10s/step × 16 = 160s），Q 估计仍不是主要瓶颈。

后续优化路径：如果 Q 估计确实成为瓶颈，可迁移到选项 A（vLLM hidden state hook）获得 KV cache 复用。但 prototype 阶段选项 C 足够。

### 2. 环境 state fork → 固定 16 个环境实例，不做 fork

每个 prompt 固定预分配 16 个环境实例，全部 `reset(item_id)` 到同一个初始状态。第 0 轮先从根状态生成 256 个候选动作，聚类后挑出 16 个代表分别执行到这 16 个实例上，形成 16 条真实分支。之后每个实例始终只沿自己的分支继续执行，不再新增实例，也不做中途 fork。

```
第 0 步: 根状态生成 256 个候选动作，全局聚类选 16 个代表 → 16 个实例各执行 1 个 action
第 1 步: 每条活跃分支各生成 16 个候选动作，做分支内聚类后只选 1 个 action 执行
第 2 步: 同理
...
```

不需要环境原生 fork 支持。每个实例从 reset 开始走自己的路径，不需要从中间状态 fork。

关键约束：每个实例每步只执行一个 action。分支内部的 16 个候选动作先聚类，再从候选代表里选 1 个执行；该动作所在簇的其余候选动作不执行环境，但保留为辅助样本。

分支终止与资源回收：
- 如果某个实例 done，该分支终止，实例空闲
- 空闲实例不再用于新 fork
- 下一步总候选动作数 = 16 × 当前活跃分支数

后续扩展：如果需要动态分支分配（某些高价值 state 分配更多实例），可以加 fork/replay 机制：
- TextCraft / BabyAI / SciWorld：state 可序列化，用 deepcopy 实现 fork
- WebArena：用 replay（从头重放 action 序列到目标 state）实现伪 fork，成本 = 当前深度 × 每步延迟
- SearchQA：搜索历史是纯文本，容易 fork

### 3. 训练数据的 batch 构造 → 选项 B：完整轨迹 + step-level advantage + 本地批结构

用最多 M=16 条被执行的完整轨迹作为 PPO 主训练数据。MClaw 内部先组织成 `TrajectoryRecord` / `ActorBatch`，再由适配层转换为外部训练后端需要的格式。

数据结构（每条轨迹）：
```
input_ids     = [prompt | s_0_obs | a_t0 | s_1_obs | a_t1 | ... | s_T_obs | a_tT]
response_mask = [0...0  | 0...0   | 1..1 | 0...0   | 1..1 | ... | 0...0   | 1..1]
advantages    = [0...0  | 0...0   | adv0 | 0...0   | adv1 | ... | 0...0   | advT]
                                     ↑                 ↑                      ↑
                            step 0 的 advantage  step 1 的 advantage   step T 的 advantage
```

其中 `adv_t` 是该 step 的 step-level advantage（标量），广播到该 turn 所有 assistant token 位置。这和 GRPO 的做法完全类似，只是从"整条轨迹一个标量"变成了"每个 turn 一个标量"。

与外部训练后端的兼容目标：
- 外部 PPO update 通常逐 token 计算 policy loss；`response_mask` 自动过滤 env observation tokens。
- MClaw 本地的 `TrajectoryRecord.response_mask` 和 `advantages` 明确保留这一语义。
- 适配层负责把本地批结构转换为外部训练后端可接受的 batch 格式。

计算效率：
- 16 条轨迹 × 平均 3500 tokens = 56,000 tokens 的 forward
- 再加同簇未执行动作对应的辅助样本；它们是短 step 级样本，不是完整轨迹
- 选项 B 中一条轨迹的 forward 自然计算了所有 turn 的 log π(a_t|s_t)，不需要重复计算 state prefix

未执行候选动作的训练信号利用：
- 同簇未执行动作进入 auxiliary loss，和被执行动作共享该 step 的 advantage / TD target
- auxiliary loss 采用簇等权：每个被选簇贡献相同总权重，簇内样本再平均，避免大簇吞掉小簇
- 其他未执行候选动作不直接进入 PPO 主训练样本
- 它们的价值主要体现在 V(s_t) 的估计上：根节点取 256 个候选动作的均值，后续每个分支取本分支 16 个候选动作的均值

### 4. 聚类方案 → hidden_state 为主方案，output_grad 为 fallback

聚类的目标分两层：
- 根节点：从 256 个候选动作中选出 16 个语义多样化的代表
- 后续分支：对每个分支自己的 16 个候选动作做分支内聚类，再从代表集中选 1 个真实执行动作

主方案: hidden_state 聚类。特征 = FSDP backbone forward 的 last-layer hidden state。

核心洞察：MClaw 的流程中，Q-critic 无论如何都要对当前 step 的全部候选动作做 FSDP forward 拿 hidden state（根节点 256 个；后续每个活跃分支 16 个）。如果聚类也用 hidden state，两者共享同一次 forward，聚类的额外计算成本 = 零。

合并后的流程：
```
1. 第 0 轮，vLLM 生成 256 个根节点候选动作
2. FSDP forward: 256 个 [state + action] → 256 个 hidden states
3. 用 hidden states 做全局聚类 → 选出 16 个代表
4. 用 hidden states 过 Q-head → 256 个 Q values
5. 16 个代表动作执行环境
6. 后续每个活跃分支重复：16 个候选动作 → 分支内聚类 → 选 1 个执行
7. 计算 step-level advantage + sibling auxiliary loss
```

hidden_state 的语义质量最高：它是模型对 (state, action) 对的完整内部表示，且和 Q-head 的输入完全对齐 — 聚类的"相似"直接意味着"Q value 相近"，确保选出的代表在 Q value 空间上多样化。

fallback 方案: output_grad 聚类。特征 = embedding(a) - softmax(logits) @ embedding_matrix。不需要额外 forward，只用 vLLM 的 top-k logprobs + embedding matrix。适用于 Q-critic 被禁用或 FSDP forward 不可用的场景。

四种方案的完整对比：

| | hidden_state | output_grad | logprob | logit_distribution |
|---|---|---|---|---|
| 语义质量 | 最高 | 中高 | 低 | 中低 |
| 额外计算 | 零（和 Q-critic 共享 forward） | 一次矩阵乘法 ~2ms | 零 | 零 |
| 需要 hack vLLM | 否（用 FSDP forward） | 否（用 top-k 近似） | 否 | 否 |
| 特征维度 | 3584（需 PCA 降维） | 3584（需 PCA 降维） | 变长（需 padding） | k=100（固定） |
| 和 Q-critic 对齐 | 完全对齐（同一特征） | 间接相关 | 无关 | 无关 |
| 推荐用途 | 主方案 | fallback | baseline / 流程验证 | 不推荐单独使用 |
