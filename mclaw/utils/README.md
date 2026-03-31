# utils

`utils/` 当前主要服务于 output-grad 聚类和 vLLM 输出解析。

## `vllm_hooks.py`

已实现的公共能力：

- `EmbeddingMatrixCache`
  - 从 actor/FSDP 模块上拿 input embedding matrix
  - 在需要时通过 `summon_full_params()` 收集完整参数
  - 支持 `warmup(refresh=True)` 显式刷新缓存

- `extract_topk_logprobs()`
  - 把 vLLM 输出规范化成：
  - `{"per_sample_topk_logprobs": ...}`

- `build_output_grad_features()`
  - 构造 `embedding(a_t) - expected_embedding(top-k)` 特征
  - 支持按 token 均值池化，或保留逐 token 特征后展平

## 当前实现细节

- top-k logprob 解析已经收紧为较明确的契约，不再试图“猜测式兼容”各种任意结构。
- `_compute_expected_embedding()` 会先对观测到的 top-k 概率质量做归一化，再计算期望 embedding。
- 当观测到的 top-k 总概率质量过低时，会发出 warning，提示 output-grad 近似可能误差较大。
- FSDP gathering 逻辑当前假设 PyTorch 2.4.0 风格的 `FullyShardedDataParallel.summon_full_params(...)` API。

## 当前状态

- utils 层已经有实装逻辑，不再只是接口文件。
- 仍未扩展到 profiling、debug dump、或更通用的 vLLM 输出适配工具。
