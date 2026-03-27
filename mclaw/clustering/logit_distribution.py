from __future__ import annotations

"""基于 top-k logit 分布的聚类器。"""

from typing import Any, Mapping, Sequence

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from .base import BaseClusterer


class LogitDistributionClusterer(BaseClusterer):
    """使用首 token 的 top-k logit 分布做聚类。"""

    def extract_features(
        self,
        action_token_ids: Sequence[Sequence[int]],
        state_token_ids: Sequence[int] | Sequence[Sequence[int]],
        model_outputs: Mapping[str, Any],
    ) -> torch.Tensor:
        """抽取可用于分布距离计算的 top-k 特征。"""
        raise NotImplementedError("TODO: 实现 logit-distribution 特征提取逻辑。")
