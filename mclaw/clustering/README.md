# clustering

`clustering/` 现在已经不是“接口占位”，而是一套可工作的 prototype 聚类实现。

## `base.py`

`BaseClusterer` 当前已经实现：

- 特征清洗和 NaN/Inf 处理
- 可选 PCA 降维
- 确定性的 torch K-Means
- 空簇重平衡
- representative 选择
- `cluster_candidates()` 对 `TreeNode` 直接聚类，并把 `cluster_id` / `cluster_feature` 回写到节点

默认距离仍然是欧氏距离。

## 各聚类器

- `action.py`
  - 直接按完整 `action token` 序列做精确分组，不依赖额外 `model_outputs`。
  - 不走 PCA / K-Means；`root_clusters` / `intra_branch_clusters` 对这个方法只作为请求值记录，不决定最终分组数。
  - 内部仍会生成一个补零后的 token feature 供调试和 `cluster_feature` 回写使用，但分组逻辑完全基于 token 序列精确匹配。
  - 适合 action 空间简单、希望验证“同动作严格视为同簇”时使用。

- `hidden_state.py`
  - 从上游 `model_outputs` 读取 hidden states，不自行触发 forward。
  - 支持 `config.hidden_state.layer` 选择目标层。
  - 如果输入是 3D hidden states，强制要求 `action_last_token_indices`，不再默默回退到序列最后一位。

- `output_grad.py`
  - 使用 `embedding(a_t) - E[top-k](embedding)` 构造特征。
  - 数据源是 `embedding_matrix` + top-k logprobs。
  - 直接复用 `mclaw.utils.vllm_hooks` 的实现，不再重复维护一套相同逻辑。

- `logprob.py`
  - 解析 token-level logprob 序列。
  - 变长 action 会被 pad 到固定维度，适合 baseline 验证流程。

- `logit_distribution.py`
  - 当前只看 action 第一个 token 的 top-k 分布。
  - 特征维度是 `top_k`，不再拼接归一化 token index。
  - 为了保留 token identity，会按 batch 内累计概率质量选择共享 token 列。
  - 仍然走欧氏 K-Means；plan 中提到的 JS/KL 距离尚未实现，所以它依然是低优先级方案。

## 当前状态

- 主流程已经可以用 `action` / `hidden_state` / `output_grad` / `logprob` / `logit_distribution` 做特征提取和聚类。
- 其中 `action` 是精确离散分组；其余向量类方法仍共用 `BaseClusterer` 的 PCA + deterministic torch K-Means。
- 仍未实现自定义距离度量、分层粗筛、或更高性能的大规模聚类后端。
