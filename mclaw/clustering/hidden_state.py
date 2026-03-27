from __future__ import annotations

"""基于 hidden state 的聚类器。"""

from typing import Any, Mapping, Sequence

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from .base import BaseClusterer


class HiddenStateClusterer(BaseClusterer):
    """使用最后一层 hidden state 作为聚类特征。"""

    def extract_features(
        self,
        action_token_ids: Sequence[Sequence[int]],
        state_token_ids: Sequence[int] | Sequence[Sequence[int]],
        model_outputs: Mapping[str, Any],
    ) -> torch.Tensor:
        """从模型输出中抽取 hidden state 特征。"""
        raise NotImplementedError("TODO: 实现 hidden-state 特征提取逻辑。")
