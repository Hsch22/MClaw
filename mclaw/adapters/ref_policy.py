from __future__ import annotations

"""对接 verl reference policy 的适配器。"""

from dataclasses import dataclass, field
from typing import Any, Mapping

from mclaw.core import ActorBatch

from .dataproto_adapter import AdaptedActorBatch, DataProtoAdapter


@dataclass(slots=True)
class VerlReferencePolicy:
    """实现 ReferencePolicyProtocol 的 verl 适配器。"""

    ref_policy: Any
    adapter: DataProtoAdapter = field(default_factory=DataProtoAdapter)
    dataproto_meta_info: dict[str, Any] = field(default_factory=dict)

    def compute_ref_log_prob(self, batch: Any) -> Mapping[str, Any]:
        source_batch, data = self._resolve_batch_and_dataproto(batch)
        data = self._move_dataproto_to_actor_device(data)
        compute_fn = getattr(self.ref_policy, "compute_ref_log_prob", None)
        if callable(compute_fn):
            raw_output = compute_fn(data)
        else:
            raw_output = self.ref_policy.compute_log_prob(data)
        payload = self.adapter.unwrap_signal_output(
            raw_output,
            preferred_keys=("ref_log_prob", "ref_log_probs", "log_probs"),
        )
        if source_batch is None:
            return {"ref_log_probs": payload}
        full_sequence_values = self.adapter.apply_signal_to_batch(
            source_batch,
            field_name="ref_log_probs",
            payload=payload,
            alignment="continuation",
        )
        return {"ref_log_probs": full_sequence_values}

    def _resolve_batch_and_dataproto(
        self,
        batch: Any,
    ) -> tuple[ActorBatch | None, Any]:
        if isinstance(batch, AdaptedActorBatch):
            return batch.source_batch, batch.dataproto

        if isinstance(batch, ActorBatch):
            data = self.adapter.to_dataproto(
                batch,
                include_ref_log_prob=False,
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

        actor_module = getattr(self.ref_policy, "actor_module", None)
        device = _resolve_module_device(actor_module)
        if device is None:
            return data
        return data.to(device)


def _resolve_module_device(module: Any) -> Any | None:
    if module is None:
        return None
    try:
        parameter = next(module.parameters())
    except (StopIteration, AttributeError, TypeError):
        return None
    return parameter.device


__all__ = ["VerlReferencePolicy"]
