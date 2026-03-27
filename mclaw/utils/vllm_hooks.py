from __future__ import annotations

"""vLLM 兼容和特征提取辅助接口。"""

from typing import Any, Mapping, Sequence

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


class EmbeddingMatrixCache:
    """缓存从 FSDP actor 收集得到的完整 embedding matrix。"""

    def __init__(self, actor_module_fsdp: Any) -> None:
        self.actor_module_fsdp = actor_module_fsdp
        self._embedding_matrix: torch.Tensor | None = None

    def warmup(self) -> None:
        """在 rollout 开始前准备 embedding matrix 缓存。"""
        raise NotImplementedError("TODO: 实现 embedding matrix all-gather 逻辑。")

    def get_embedding_matrix(self) -> torch.Tensor:
        """返回完整 vocab 的 embedding matrix。"""
        raise NotImplementedError("TODO: 实现 embedding matrix 读取逻辑。")


def extract_topk_logprobs(generation_output: Any) -> Mapping[str, Any]:
    """从 vLLM 输出中抽取 top-k logprob 结构。"""
    raise NotImplementedError("TODO: 实现 top-k logprob 抽取逻辑。")


def build_output_grad_features(
    action_token_ids: Sequence[Sequence[int]],
    topk_logprobs: Mapping[str, Any],
    embedding_matrix: torch.Tensor,
    use_mean_pooling: bool = True,
) -> torch.Tensor:
    """把 output-gradient 近似整理为聚类特征。"""
    raise NotImplementedError("TODO: 实现 output-gradient 特征构造逻辑。")
