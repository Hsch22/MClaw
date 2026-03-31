from __future__ import annotations

"""MClaw <-> verl.DataProto 的核心适配层。"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Any

from mclaw.core import ActorBatch, TrajectoryRecord


@dataclass(slots=True)
class _TrajectoryTensorView:
    input_ids: list[int]
    attention_mask: list[int]
    position_ids: list[int]
    responses: list[int]
    response_mask: list[int]
    advantages: list[float]
    old_log_probs: list[float]
    ref_log_prob: list[float] | None
    prompt_length: int


@dataclass(slots=True)
class AdaptedActorBatch:
    """保留源 `ActorBatch` 和已构造 `DataProto` 的轻量包装。"""

    source_batch: ActorBatch
    dataproto: Any


@dataclass(slots=True)
class DataProtoAdapter:
    """把 MClaw 的 ActorBatch 映射成 verl.DataProto。"""

    pad_token_id: int = 0
    default_meta_info: dict[str, Any] = field(default_factory=dict)

    def to_dataproto(
        self,
        batch: ActorBatch,
        *,
        include_ref_log_prob: bool = False,
        meta_info_overrides: Mapping[str, Any] | None = None,
    ) -> Any:
        """将 ActorBatch 转成 verl.DataProto。

        注意这里的 `responses` 使用的是 `prompt_length` 之后的整段 continuation，
        而不是 `TrajectoryRecord.responses`。后者在当前 MClaw 实现里只包含 action
        token，会丢掉多轮轨迹中的 observation token。
        """
        if not batch.trajectories:
            raise ValueError("ActorBatch must contain at least one trajectory")

        torch = _import_torch()
        DataProto = _import_dataproto()
        TensorDict = _import_tensordict()
        np = _import_numpy()

        views = [
            self._build_trajectory_view(
                trajectory,
                include_ref_log_prob=include_ref_log_prob,
            )
            for trajectory in batch.trajectories
        ]

        tensors: dict[str, Any] = {
            "input_ids": self._pad_int_sequences(
                [view.input_ids for view in views],
                pad_value=self.pad_token_id,
                torch_module=torch,
            ),
            "attention_mask": self._pad_int_sequences(
                [view.attention_mask for view in views],
                pad_value=0,
                torch_module=torch,
            ),
            "position_ids": self._pad_int_sequences(
                [view.position_ids for view in views],
                pad_value=0,
                torch_module=torch,
            ),
            "responses": self._pad_int_sequences(
                [view.responses for view in views],
                pad_value=self.pad_token_id,
                torch_module=torch,
            ),
            "response_mask": self._pad_int_sequences(
                [view.response_mask for view in views],
                pad_value=0,
                torch_module=torch,
            ),
            "advantages": self._pad_float_sequences(
                [view.advantages for view in views],
                pad_value=0.0,
                torch_module=torch,
            ),
            "old_log_probs": self._pad_float_sequences(
                [view.old_log_probs for view in views],
                pad_value=0.0,
                torch_module=torch,
            ),
        }

        if include_ref_log_prob:
            tensors["ref_log_prob"] = self._pad_float_sequences(
                [view.ref_log_prob or [] for view in views],
                pad_value=0.0,
                torch_module=torch,
            )

        meta_info = self._resolve_meta_info(
            batch=batch,
            views=views,
            overrides=meta_info_overrides,
        )
        non_tensors = {
            "trajectory_metadata": np.array(
                [dict(record.metadata) for record in batch.trajectories],
                dtype=object,
            ),
        }
        tensor_batch = TensorDict(
            source=tensors,
            batch_size=[len(batch.trajectories)],
        )
        return DataProto(
            batch=tensor_batch,
            non_tensor_batch=non_tensors,
            meta_info=meta_info,
        )

    def adapt_actor_batch(
        self,
        batch: ActorBatch,
        *,
        include_ref_log_prob: bool = False,
        meta_info_overrides: Mapping[str, Any] | None = None,
    ) -> AdaptedActorBatch:
        """构造同时携带源 batch 和 `DataProto` 的适配结果。"""
        return AdaptedActorBatch(
            source_batch=batch,
            dataproto=self.to_dataproto(
                batch,
                include_ref_log_prob=include_ref_log_prob,
                meta_info_overrides=meta_info_overrides,
            ),
        )

    def unwrap_signal_output(
        self,
        output: Any,
        *,
        preferred_keys: Sequence[str],
    ) -> Any:
        """从 tensor / mapping / DataProto-like 输出中提取主信号。"""
        if isinstance(output, Mapping):
            for key in preferred_keys:
                if key in output:
                    return output[key]
            if len(output) == 1:
                return next(iter(output.values()))
            return output

        batch = getattr(output, "batch", None)
        keys = getattr(batch, "keys", None)
        if callable(keys):
            available_keys = set(keys())
            for key in preferred_keys:
                if key in available_keys:
                    return batch[key]

        return output

    def expand_signal_to_full_sequences(
        self,
        batch: ActorBatch,
        payload: Any,
    ) -> list[list[float]]:
        """把 response-only / continuation-only 信号扩展回 full sequence。"""
        rows = self._coerce_rows(payload)
        if len(rows) != len(batch.trajectories):
            raise ValueError(
                "Signal payload batch size does not match ActorBatch: "
                f"{len(rows)} vs {len(batch.trajectories)}"
            )

        expanded: list[list[float]] = []
        for record, row in zip(batch.trajectories, rows):
            full_length = len(record.input_ids)
            prompt_length = self.infer_prompt_length(record)
            continuation_length = full_length - prompt_length
            response_token_count = sum(int(flag) for flag in record.response_mask)

            if len(row) == full_length:
                expanded.append(list(row))
                continue

            if len(row) == continuation_length:
                expanded.append(([0.0] * prompt_length) + list(row))
                continue

            if len(row) == response_token_count:
                values = [0.0] * full_length
                response_index = 0
                for token_index, is_response in enumerate(record.response_mask):
                    if not is_response:
                        continue
                    values[token_index] = row[response_index]
                    response_index += 1
                expanded.append(values)
                continue

            raise ValueError(
                "Signal row length does not match trajectory layout: "
                f"full={full_length}, continuation={continuation_length}, "
                f"response={response_token_count}, got={len(row)}"
            )

        return expanded

    def apply_signal_to_batch(
        self,
        batch: ActorBatch,
        *,
        field_name: str,
        payload: Any,
    ) -> list[list[float]]:
        """把后端输出写回每条 TrajectoryRecord。"""
        expanded = self.expand_signal_to_full_sequences(batch, payload)
        for record, values in zip(batch.trajectories, expanded):
            setattr(record, field_name, list(values))
        batch.metadata[field_name] = expanded
        return expanded

    def infer_prompt_length(self, record: TrajectoryRecord) -> int:
        """推断多轮轨迹里初始 prompt 的长度。"""
        metadata_prompt_length = record.metadata.get("prompt_length")
        if metadata_prompt_length is not None:
            try:
                prompt_length = int(metadata_prompt_length)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid prompt_length in trajectory metadata: {metadata_prompt_length!r}"
                ) from exc
            if prompt_length < 0 or prompt_length > len(record.input_ids):
                raise ValueError(
                    f"prompt_length out of range: {prompt_length} for "
                    f"sequence length {len(record.input_ids)}"
                )
            return prompt_length

        if record.steps:
            prompt_length = len(record.steps[0].state_tokens)
            if prompt_length > len(record.input_ids):
                raise ValueError(
                    f"Derived prompt_length {prompt_length} exceeds sequence length "
                    f"{len(record.input_ids)}"
                )
            return prompt_length

        return len(record.input_ids)

    def _build_trajectory_view(
        self,
        record: TrajectoryRecord,
        *,
        include_ref_log_prob: bool,
    ) -> _TrajectoryTensorView:
        input_ids = list(record.input_ids)
        prompt_length = self.infer_prompt_length(record)
        continuation_length = len(input_ids) - prompt_length

        attention_mask = self._resolve_int_field(
            values=record.attention_mask,
            expected_length=len(input_ids),
            field_name="attention_mask",
            default_value=1,
        )
        position_ids = self._resolve_position_ids(record)
        response_mask = self._resolve_int_field(
            values=record.response_mask,
            expected_length=len(input_ids),
            field_name="response_mask",
            default_value=0,
        )[prompt_length:]
        advantages = self._resolve_float_field(
            values=record.advantages,
            expected_length=len(input_ids),
            field_name="advantages",
            default_value=0.0,
        )[prompt_length:]
        old_log_probs = self._resolve_signal_field(
            values=record.old_log_probs,
            record=record,
            field_name="old_log_probs",
        )[prompt_length:]

        ref_log_prob: list[float] | None = None
        if include_ref_log_prob:
            ref_log_prob = self._resolve_signal_field(
                values=record.ref_log_probs,
                record=record,
                field_name="ref_log_probs",
                allow_empty=False,
            )[prompt_length:]

        if len(response_mask) != continuation_length:
            raise ValueError(
                "response_mask continuation length mismatch: "
                f"{len(response_mask)} vs {continuation_length}"
            )
        if len(advantages) != continuation_length:
            raise ValueError(
                "advantages continuation length mismatch: "
                f"{len(advantages)} vs {continuation_length}"
            )
        if len(old_log_probs) != continuation_length:
            raise ValueError(
                "old_log_probs continuation length mismatch: "
                f"{len(old_log_probs)} vs {continuation_length}"
            )
        if ref_log_prob is not None and len(ref_log_prob) != continuation_length:
            raise ValueError(
                "ref_log_prob continuation length mismatch: "
                f"{len(ref_log_prob)} vs {continuation_length}"
            )

        return _TrajectoryTensorView(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            responses=input_ids[prompt_length:],
            response_mask=response_mask,
            advantages=advantages,
            old_log_probs=old_log_probs,
            ref_log_prob=ref_log_prob,
            prompt_length=prompt_length,
        )

    def _resolve_meta_info(
        self,
        *,
        batch: ActorBatch,
        views: Sequence[_TrajectoryTensorView],
        overrides: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        meta_info = dict(self.default_meta_info)
        meta_info.update(batch.metadata.get("dataproto_meta_info", {}))
        if overrides:
            meta_info.update(dict(overrides))

        if "micro_batch_size" not in meta_info:
            meta_info["micro_batch_size"] = len(batch.trajectories)
        if "temperature" not in meta_info:
            meta_info["temperature"] = 1.0
        if "use_dynamic_bsz" not in meta_info:
            meta_info["use_dynamic_bsz"] = False
        if "pad_token_id" not in meta_info:
            meta_info["pad_token_id"] = self.pad_token_id
        if meta_info.get("use_dynamic_bsz") and "max_token_len" not in meta_info:
            meta_info["max_token_len"] = max((len(view.input_ids) for view in views), default=0)

        return meta_info

    def _resolve_position_ids(self, record: TrajectoryRecord) -> list[int]:
        if not record.position_ids:
            return list(range(len(record.input_ids)))
        if len(record.position_ids) != len(record.input_ids):
            raise ValueError(
                "position_ids length mismatch: "
                f"{len(record.position_ids)} vs {len(record.input_ids)}"
            )
        return [int(value) for value in record.position_ids]

    def _resolve_int_field(
        self,
        *,
        values: Sequence[Any],
        expected_length: int,
        field_name: str,
        default_value: int,
    ) -> list[int]:
        if not values:
            return [default_value] * expected_length
        if len(values) != expected_length:
            raise ValueError(
                f"{field_name} length mismatch: {len(values)} vs {expected_length}"
            )
        return [int(value) for value in values]

    def _resolve_float_field(
        self,
        *,
        values: Sequence[Any],
        expected_length: int,
        field_name: str,
        default_value: float,
    ) -> list[float]:
        if not values:
            return [default_value] * expected_length
        if len(values) != expected_length:
            raise ValueError(
                f"{field_name} length mismatch: {len(values)} vs {expected_length}"
            )
        return [self._to_float(value, field_name=field_name) for value in values]

    def _resolve_signal_field(
        self,
        *,
        values: Sequence[Any],
        record: TrajectoryRecord,
        field_name: str,
        allow_empty: bool = True,
    ) -> list[float]:
        expected_length = len(record.input_ids)
        prompt_length = self.infer_prompt_length(record)
        continuation_length = expected_length - prompt_length
        response_token_count = sum(int(flag) for flag in record.response_mask)

        if not values:
            if allow_empty:
                return [0.0] * expected_length
            raise ValueError(f"{field_name} is required but missing")

        if len(values) == expected_length:
            return [self._to_float(value, field_name=field_name) for value in values]

        if len(values) == continuation_length:
            return ([0.0] * prompt_length) + [
                self._to_float(value, field_name=field_name) for value in values
            ]

        if len(values) == response_token_count:
            resolved = [0.0] * expected_length
            response_index = 0
            for token_index, is_response in enumerate(record.response_mask):
                if not is_response:
                    continue
                resolved[token_index] = self._to_float(
                    values[response_index],
                    field_name=field_name,
                )
                response_index += 1
            return resolved

        raise ValueError(
            f"{field_name} length does not match trajectory layout: "
            f"full={expected_length}, continuation={continuation_length}, "
            f"response={response_token_count}, got={len(values)}"
        )

    def _coerce_rows(self, payload: Any) -> list[list[float]]:
        rows = self._to_sequence(payload)
        if rows is None:
            raise TypeError("signal payload must be sequence-like or expose tolist()")

        normalized: list[list[float]] = []
        for row in rows:
            row_values = self._to_sequence(row)
            if row_values is None:
                raise TypeError("signal payload rows must be sequence-like")
            normalized.append(
                [self._to_float(value, field_name="signal_payload") for value in row_values]
            )
        return normalized

    def _pad_int_sequences(
        self,
        rows: Sequence[Sequence[int]],
        *,
        pad_value: int,
        torch_module: Any,
    ) -> Any:
        max_length = max((len(row) for row in rows), default=0)
        padded = [
            [int(value) for value in row] + ([int(pad_value)] * (max_length - len(row)))
            for row in rows
        ]
        return torch_module.tensor(padded, dtype=torch_module.long)

    def _pad_float_sequences(
        self,
        rows: Sequence[Sequence[float]],
        *,
        pad_value: float,
        torch_module: Any,
    ) -> Any:
        max_length = max((len(row) for row in rows), default=0)
        padded = [
            [float(value) for value in row] + ([float(pad_value)] * (max_length - len(row)))
            for row in rows
        ]
        return torch_module.tensor(padded, dtype=torch_module.float32)

    def _to_sequence(self, value: Any) -> list[Any] | None:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return list(value)
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            try:
                converted = tolist()
            except (TypeError, ValueError):
                return None
            if isinstance(converted, Sequence) and not isinstance(
                converted,
                (str, bytes, bytearray),
            ):
                return list(converted)
        return None

    def _to_float(self, value: Any, *, field_name: str) -> float:
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, Real):
            return float(value)
        item = getattr(value, "item", None)
        if callable(item):
            try:
                scalar = item()
            except (TypeError, ValueError) as exc:
                raise TypeError(f"{field_name} contains non-scalar tensor-like value") from exc
            if isinstance(scalar, bool):
                return float(scalar)
            if isinstance(scalar, Real):
                return float(scalar)
        raise TypeError(f"{field_name} contains non-numeric value: {value!r}")


def _import_torch() -> Any:
    import torch

    return torch


def _import_dataproto() -> Any:
    from verl import DataProto

    return DataProto


def _import_tensordict() -> Any:
    from tensordict import TensorDict

    return TensorDict


def _import_numpy() -> Any:
    import numpy as np

    return np


__all__ = ["AdaptedActorBatch", "DataProtoAdapter"]
