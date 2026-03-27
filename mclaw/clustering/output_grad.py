from __future__ import annotations

"""基于 output-gradient 近似的聚类器。"""

from typing import Any, Mapping, Sequence

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from .base import BaseClusterer


class OutputGradClusterer(BaseClusterer):
    """使用 output-layer 梯度近似构造聚类特征。"""

    def extract_features(
        self,
        action_token_ids: Sequence[Sequence[int]],
        state_token_ids: Sequence[int] | Sequence[Sequence[int]],
        model_outputs: Mapping[str, Any],
    ) -> torch.Tensor:
        """根据 top-k logprobs 和 embedding matrix 构造特征。"""
        raise NotImplementedError("TODO: 实现 output-grad 特征提取逻辑。")

    def build_expected_embeddings(
        self,
        topk_logprobs: Mapping[str, Any],
        embedding_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """计算策略期望 embedding。"""
        raise NotImplementedError("TODO: 实现期望 embedding 计算逻辑。")
