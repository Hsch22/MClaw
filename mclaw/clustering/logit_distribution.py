from __future__ import annotations

"""基于 top-k logit 分布的聚类器。"""

import math
from collections.abc import Mapping as MappingABC, Sequence as SequenceABC
from typing import Any, Mapping, Sequence

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from mclaw.utils.vllm_hooks import coerce_per_sample_topk_logprobs, extract_topk_logprobs

from .base import BaseClusterer, resolve_model_output_field


class LogitDistributionClusterer(BaseClusterer):
    """使用 action 首 token 的 top-k logprob 分布做聚类。"""

    def extract_features(
        self,
        action_token_ids: Sequence[Sequence[int]],
        state_token_ids: Sequence[int] | Sequence[Sequence[int]],
        model_outputs: Mapping[str, Any],
    ) -> torch.Tensor:
        """抽取首 token 的固定维度 top-k 分布特征。

        当前实现是一个低优先级占位方案：
        - 数据源复用 vLLM generate 阶段的 top-k logprobs，零额外 forward 成本
        - 只看 action 第一个 token 的分布，维度固定为 top_k
        - 为了在固定维度里保留 token identity，会按 batch 内累计概率质量选择共享 token 列
        - 仍复用 BaseClusterer 的欧氏 K-Means；plan 中提到更合适的 JS/KL 距离尚未实现
        """
        del state_token_ids
        self._require_torch()

        per_sample_topk = _resolve_per_sample_topk_logprobs(model_outputs)
        if len(per_sample_topk) != len(action_token_ids):
            raise ValueError(
                "top-k logprob sample count does not match action count: "
                f"{len(per_sample_topk)} != {len(action_token_ids)}"
            )

        top_k = max(int(self.config.logit_distribution.top_k), 1)
        first_position_distributions = [
            _normalize_first_position_distribution(_resolve_first_position_topk(per_position_topk))
            for per_position_topk in per_sample_topk
        ]
        selected_token_ids = _select_feature_token_ids(first_position_distributions, top_k)
        features = torch.zeros((len(action_token_ids), top_k), dtype=torch.float32)
        if not selected_token_ids:
            return features

        token_to_column = {token_id: column for column, token_id in enumerate(selected_token_ids)}
        for row_index, distribution in enumerate(first_position_distributions):
            for token_id, probability in distribution.items():
                column = token_to_column.get(token_id)
                if column is None:
                    continue
                features[row_index, column] = probability
        return features


def _resolve_per_sample_topk_logprobs(model_outputs: Mapping[str, Any]) -> list[Any]:
    topk_logprobs = resolve_model_output_field(model_outputs, "topk_logprobs")
    if topk_logprobs is not None:
        return coerce_per_sample_topk_logprobs(topk_logprobs, source_name="topk_logprobs")

    generation_output = resolve_model_output_field(model_outputs, "generation_output")
    extracted = extract_topk_logprobs(generation_output if generation_output is not None else model_outputs)
    return coerce_per_sample_topk_logprobs(
        extracted,
        source_name="extract_topk_logprobs(...)",
    )


def _resolve_first_position_topk(per_position_topk: Any) -> Mapping[int, float]:
    if isinstance(per_position_topk, SequenceABC) and not isinstance(per_position_topk, (str, bytes, bytearray)):
        if not per_position_topk:
            return {}
        first_position = per_position_topk[0]
        if isinstance(first_position, MappingABC):
            return first_position
        raise TypeError("first-position top-k logprobs must be a mapping")
    raise TypeError("per-sample top-k logprobs must be a sequence of per-position mappings")


def _normalize_first_position_distribution(first_position: Mapping[int, float]) -> dict[int, float]:
    normalized: dict[int, float] = {}
    for token_id, logprob in first_position.items():
        try:
            coerced_token_id = int(token_id)
            coerced_logprob = float(logprob)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(coerced_logprob):
            continue
        normalized[coerced_token_id] = math.exp(coerced_logprob)
    return normalized


def _select_feature_token_ids(
    first_position_distributions: Sequence[Mapping[int, float]],
    top_k: int,
) -> list[int]:
    token_mass: dict[int, float] = {}
    for distribution in first_position_distributions:
        for token_id, probability in distribution.items():
            token_mass[token_id] = token_mass.get(token_id, 0.0) + float(probability)

    ranked = sorted(token_mass.items(), key=lambda item: (-item[1], item[0]))
    return [token_id for token_id, _ in ranked[:top_k]]
