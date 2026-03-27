from __future__ import annotations

"""基于 logprob 向量的基线聚类器。"""

from typing import Any, Mapping, Sequence

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from .base import BaseClusterer


class LogProbClusterer(BaseClusterer):
    """使用 action token logprob 向量做聚类。"""

    def extract_features(
        self,
        action_token_ids: Sequence[Sequence[int]],
        state_token_ids: Sequence[int] | Sequence[Sequence[int]],
        model_outputs: Mapping[str, Any],
    ) -> torch.Tensor:
        """把变长 logprob 序列整理成固定维度特征。"""
        raise NotImplementedError("TODO: 实现 logprob 特征提取逻辑。")
