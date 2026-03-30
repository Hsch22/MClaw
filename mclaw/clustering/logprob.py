from __future__ import annotations

"""基于 logprob 向量的基线聚类器。"""

from collections.abc import Mapping as MappingABC, Sequence as SequenceABC
from numbers import Real
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
        del state_token_ids
        self._require_torch()

        raw_sequences = self._resolve_logprob_sequences(model_outputs)
        if len(raw_sequences) != len(action_token_ids):
            raise ValueError(
                "logprob sequence count does not match action count: "
                f"{len(raw_sequences)} != {len(action_token_ids)}"
            )

        normalized_sequences: list[list[float]] = []
        max_length = 0
        for action_ids, raw_sequence in zip(action_token_ids, raw_sequences):
            expected_length = len(action_ids)
            sequence = self._normalize_logprob_sequence(raw_sequence, action_ids)
            if expected_length > 0:
                if len(sequence) < expected_length:
                    sequence = sequence + [0.0] * (expected_length - len(sequence))
                else:
                    sequence = sequence[:expected_length]

            normalized_sequences.append(sequence)
            max_length = max(max_length, len(sequence))

        # 保证返回合法的 2D feature matrix。
        feature_length = max(max_length, 1)
        features = torch.zeros(
            (len(normalized_sequences), feature_length),
            dtype=torch.float32,
        )
        for row_index, sequence in enumerate(normalized_sequences):
            if not sequence:
                continue
            features[row_index, : len(sequence)] = torch.tensor(sequence, dtype=torch.float32)
        return features

    def _resolve_logprob_sequences(self, model_outputs: Mapping[str, Any]) -> list[Any]:
        """从若干常见字段中抽取按样本组织的 logprob 序列。"""
        for key in ("action_logprobs", "token_logprobs", "logprobs"):
            if key in model_outputs:
                return self._coerce_sample_sequences(model_outputs[key])

        outputs = model_outputs.get("outputs")
        if outputs is not None:
            sample_sequences: list[Any] = []
            for output in self._coerce_sample_sequences(outputs):
                sample_sequences.append(self._extract_logprobs_from_output(output))
            return sample_sequences

        available_keys = ", ".join(sorted(model_outputs.keys()))
        raise KeyError(
            "model_outputs must contain one of "
            "'action_logprobs', 'token_logprobs', 'logprobs', or 'outputs'; "
            f"got keys: [{available_keys}]"
        )

    def _extract_logprobs_from_output(self, output: Any) -> Any:
        """从单个生成输出对象中抽取 logprob 序列。"""
        if isinstance(output, MappingABC):
            for key in ("action_logprobs", "token_logprobs", "logprobs"):
                if key in output:
                    return output[key]
        for attr in ("action_logprobs", "token_logprobs", "logprobs"):
            if hasattr(output, attr):
                return getattr(output, attr)
        raise KeyError("failed to extract logprobs from generation output item")

    def _coerce_sample_sequences(self, value: Any) -> list[Any]:
        """把不同容器格式统一成按样本组织的列表。"""
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                return [[float(value.item())]]
            if value.ndim == 1:
                return [value]
            return [value[index] for index in range(value.size(0))]

        if isinstance(value, SequenceABC) and not isinstance(value, (str, bytes, bytearray)):
            return list(value)

        raise TypeError(f"unsupported logprob container type: {type(value)!r}")

    def _normalize_logprob_sequence(
        self,
        sequence: Any,
        action_token_ids: Sequence[int],
    ) -> list[float]:
        """将单个样本的 token-level logprob 结构拍平成 float 列表。"""
        if isinstance(sequence, torch.Tensor):
            if sequence.ndim == 0:
                return [float(sequence.item())]
            return [float(item) for item in sequence.detach().float().reshape(-1).tolist()]

        if not isinstance(sequence, SequenceABC) or isinstance(sequence, (str, bytes, bytearray)):
            return [self._extract_position_logprob(sequence, action_token_ids[0] if action_token_ids else None)]

        normalized: list[float] = []
        for position, item in enumerate(sequence):
            token_id = action_token_ids[position] if position < len(action_token_ids) else None
            normalized.append(self._extract_position_logprob(item, token_id))
        return normalized

    def _extract_position_logprob(self, value: Any, token_id: int | None) -> float:
        """从一个 token 位置的输出对象里取出对应 logprob。"""
        if isinstance(value, Real):
            return float(value)

        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                return float(value.item())
            flattened = value.detach().float().reshape(-1)
            if flattened.numel() == 0:
                return 0.0
            return float(flattened[0].item())

        if hasattr(value, "logprob"):
            return float(getattr(value, "logprob"))

        if isinstance(value, MappingABC):
            if token_id is not None:
                for key in (token_id, str(token_id)):
                    if key in value:
                        return self._extract_position_logprob(value[key], None)
            if "logprob" in value:
                return self._extract_position_logprob(value["logprob"], None)
            if len(value) == 1:
                return self._extract_position_logprob(next(iter(value.values())), None)
            raise KeyError(
                f"failed to resolve token logprob for token_id={token_id} from mapping output"
            )

        if isinstance(value, SequenceABC) and not isinstance(value, (str, bytes, bytearray)):
            if len(value) == 0:
                return 0.0
            return self._extract_position_logprob(value[0], token_id)

        raise TypeError(f"unsupported token logprob value type: {type(value)!r}")
