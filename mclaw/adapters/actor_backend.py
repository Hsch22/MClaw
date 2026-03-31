from __future__ import annotations

"""对接 verl DataParallelPPOActor 的 actor backend。"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Any

from mclaw.core import ActorBatch, AuxiliaryBatch

from .dataproto_adapter import AdaptedActorBatch, DataProtoAdapter


@dataclass(slots=True)
class VerlActorBackend:
    """实现 ActorBackendProtocol 的 verl 适配器。"""

    actor: Any
    adapter: DataProtoAdapter = field(default_factory=DataProtoAdapter)
    dataproto_meta_info: dict[str, Any] = field(default_factory=dict)
    use_kl_loss: bool | None = None

    def compute_log_prob(self, batch: Any) -> Mapping[str, Any]:
        """重算 old log-probs，并扩展回 full-sequence 形状。"""
        source_batch, data = self._resolve_batch_and_dataproto(
            batch,
            include_ref_log_prob=False,
        )
        data = self._move_dataproto_to_actor_device(data)
        raw_output = self.actor.compute_log_prob(data)
        payload = self.adapter.unwrap_signal_output(
            raw_output,
            preferred_keys=("old_log_probs", "log_probs"),
        )
        if source_batch is None:
            return {"old_log_probs": payload}
        full_sequence_values = self.adapter.apply_signal_to_batch(
            source_batch,
            field_name="old_log_probs",
            payload=payload,
        )
        return {"old_log_probs": full_sequence_values}

    def update_policy(self, batch: Any) -> Mapping[str, float]:
        """执行 PPO update，并把后端 list-metrics 规约成标量。"""
        _, data = self._resolve_batch_and_dataproto(
            batch,
            include_ref_log_prob=self._should_include_ref_log_prob(),
        )
        raw_output = self.actor.update_policy(data)
        metrics = self._extract_metrics_mapping(raw_output)
        return self._reduce_metrics(metrics)

    def update_aux_loss(self, batch: AuxiliaryBatch) -> Mapping[str, float]:
        """当前 prototype 默认把 auxiliary loss 叠加到主 PPO loss。"""
        del batch
        return {}

    def _should_include_ref_log_prob(self) -> bool:
        if self.use_kl_loss is not None:
            return bool(self.use_kl_loss)
        return bool(_resolve_nested_value(self.actor, ("config", "use_kl_loss"), default=False))

    def _resolve_batch_and_dataproto(
        self,
        batch: Any,
        *,
        include_ref_log_prob: bool,
    ) -> tuple[ActorBatch | None, Any]:
        if isinstance(batch, AdaptedActorBatch):
            return batch.source_batch, batch.dataproto

        if isinstance(batch, ActorBatch):
            data = self.adapter.to_dataproto(
                batch,
                include_ref_log_prob=include_ref_log_prob,
                meta_info_overrides=self.dataproto_meta_info,
            )
            return batch, data

        if hasattr(batch, "batch") and hasattr(batch, "meta_info"):
            return None, batch

        raise TypeError(
            "batch must be an ActorBatch, an AdaptedActorBatch, or a DataProto-like object"
        )

    def _move_dataproto_to_actor_device(self, data: Any) -> Any:
        to_device = getattr(data, "to", None)
        if not callable(to_device):
            return data

        actor_module = getattr(self.actor, "actor_module", None)
        device = _resolve_module_device(actor_module)
        if device is None:
            return data
        return data.to(device)

    def _extract_metrics_mapping(self, output: Any) -> Mapping[str, Any]:
        if output is None:
            return {}

        if isinstance(output, Mapping):
            return output

        meta_info = getattr(output, "meta_info", None)
        if isinstance(meta_info, Mapping):
            metrics = meta_info.get("metrics")
            if isinstance(metrics, Mapping):
                return metrics
            return {}

        raise TypeError(
            "actor.update_policy() must return a metrics mapping or a DataProto-like "
            "object with meta_info['metrics']"
        )

    def _reduce_metrics(self, metrics: Mapping[str, Any]) -> dict[str, float]:
        reduced: dict[str, float] = {}
        for key, value in metrics.items():
            metric_value = self._reduce_metric_value(value)
            if metric_value is None:
                continue
            reduced[key] = metric_value
        return reduced

    def _reduce_metric_value(self, value: Any) -> float | None:
        scalar = self._to_float(value)
        if scalar is not None:
            return scalar

        sequence = self._to_sequence(value)
        if sequence is None or not sequence:
            return None

        total = 0.0
        count = 0
        for item in sequence:
            item_scalar = self._to_float(item)
            if item_scalar is None:
                continue
            total += item_scalar
            count += 1
        if count == 0:
            return None
        return total / count

    def _to_float(self, value: Any) -> float | None:
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, Real):
            return float(value)
        item = getattr(value, "item", None)
        if callable(item):
            try:
                scalar = item()
            except (TypeError, ValueError):
                return None
            if isinstance(scalar, bool):
                return float(scalar)
            if isinstance(scalar, Real):
                return float(scalar)
        return None

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


def _resolve_nested_value(value: Any, path: tuple[str, ...], default: Any) -> Any:
    current = value
    for key in path:
        if current is None:
            return default
        if isinstance(current, Mapping):
            if key not in current:
                return default
            current = current[key]
            continue
        if hasattr(current, key):
            current = getattr(current, key)
            continue
        getter = getattr(current, "get", None)
        if callable(getter):
            sentinel = object()
            try:
                candidate = getter(key, sentinel)
            except TypeError:
                try:
                    candidate = getter(key)
                except Exception:
                    return default
            except Exception:
                return default
            if candidate is sentinel:
                return default
            current = candidate
            continue
        return default
    return current


def _resolve_module_device(module: Any) -> Any | None:
    if module is None:
        return None
    try:
        parameter = next(module.parameters())
    except (StopIteration, AttributeError, TypeError):
        return None
    return parameter.device


__all__ = ["VerlActorBackend"]
