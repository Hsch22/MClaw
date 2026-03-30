from __future__ import annotations

"""基于 output-gradient 近似的聚类器。"""

from typing import Any, Mapping, Sequence

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from mclaw.utils.vllm_hooks import build_output_grad_features, extract_topk_logprobs

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
        del state_token_ids
        self._require_torch()

        embedding_matrix = model_outputs.get("embedding_matrix")
        if embedding_matrix is None:
            raise KeyError("model_outputs must provide embedding_matrix for output-grad clustering")

        topk_logprobs = model_outputs.get("topk_logprobs")
        if topk_logprobs is None:
            topk_logprobs = extract_topk_logprobs(model_outputs.get("generation_output", model_outputs))

        return build_output_grad_features(
            action_token_ids=action_token_ids,
            topk_logprobs=topk_logprobs,
            embedding_matrix=embedding_matrix,
            use_mean_pooling=self.config.output_grad.use_mean_pooling,
        )
