# utils

`utils/` 存放不属于主算法模块的辅助接口。

## 当前文件

- `__init__.py`：导出 utils 公共接口。
- `vllm_hooks.py`：定义 vLLM 兼容和特征提取辅助接口。
  - `EmbeddingMatrixCache`
  - `extract_topk_logprobs()`
  - `build_output_grad_features()`

## 当前职责

- 管理 output-grad 聚类所需的 embedding matrix 缓存
- 从推理输出中抽取 top-k logprob 结构
- 构造 output-gradient 近似特征

## 当前状态

- 目前只建立了接口层。
- 分布式辅助函数、数据转换工具、调试和 profile 工具尚未补充。
