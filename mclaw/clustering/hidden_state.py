from __future__ import annotations

"""基于 hidden state 的聚类器。"""

from typing import Any, Mapping, Sequence

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from .base import BaseClusterer, resolve_model_output_field


class HiddenStateClusterer(BaseClusterer):
    """使用最后一层 hidden state 作为聚类特征。"""

    def extract_features(
        self,
        action_token_ids: Sequence[Sequence[int]],
        state_token_ids: Sequence[int] | Sequence[Sequence[int]],
        model_outputs: Mapping[str, Any],
    ) -> torch.Tensor:
        """从上游 FSDP forward 产出的模型输出中抽取 hidden state 特征。"""
        # Hidden-state 聚类完全复用上游已经算好的 FSDP forward 结果，本方法不触发任何前向计算。
        self._require_torch()

        token_hidden_states = resolve_model_output_field(model_outputs, "token_hidden_states")
        if token_hidden_states is not None:
            return self._pool_action_token_hidden_states(
                token_hidden_states=token_hidden_states,
                action_token_ids=action_token_ids,
                state_token_ids=state_token_ids,
            )

        hidden_states = resolve_model_output_field(model_outputs, "hidden_states")
        if hidden_states is None:
            raise KeyError("model_outputs must provide hidden_states for hidden-state clustering")
        hidden_states = self._select_hidden_states(hidden_states)

        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError("resolved hidden_states must be a torch.Tensor")

        if hidden_states.ndim == 2:
            return hidden_states.detach().float()

        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must be 2D or 3D, "
                f"got shape {tuple(hidden_states.shape)}"
            )

        action_last_token_indices = resolve_model_output_field(model_outputs, "action_last_token_indices")
        if action_last_token_indices is None:
            raise KeyError(
                "model_outputs must provide action_last_token_indices when hidden_states is 3D; "
                "falling back to the last sequence position is unsafe under padded batching"
            )

        indices = torch.tensor(list(action_last_token_indices), dtype=torch.long, device=hidden_states.device)
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_indices, indices].detach().float()

    def _pool_action_token_hidden_states(
        self,
        *,
        token_hidden_states: Any,
        action_token_ids: Sequence[Sequence[int]],
        state_token_ids: Sequence[int] | Sequence[Sequence[int]],
    ) -> torch.Tensor:
        pooling = str(getattr(self.config.hidden_state, "token_pooling", "last")).strip().lower()
        last_k = max(int(getattr(self.config.hidden_state, "last_k", 4)), 1)
        batch_state_token_ids = self._normalize_state_token_ids(state_token_ids, len(action_token_ids))

        if isinstance(token_hidden_states, torch.Tensor):
            if token_hidden_states.ndim != 3:
                raise ValueError(
                    "token_hidden_states must be 3D when provided as a tensor, "
                    f"got shape {tuple(token_hidden_states.shape)}"
                )
            per_sample_hidden_states = [token_hidden_states[index] for index in range(token_hidden_states.size(0))]
        elif isinstance(token_hidden_states, Sequence):
            per_sample_hidden_states = list(token_hidden_states)
        else:
            raise TypeError(
                "token_hidden_states must be a candidate-aligned sequence or 3D tensor, "
                f"got {type(token_hidden_states).__name__}"
            )

        if len(per_sample_hidden_states) != len(action_token_ids):
            raise ValueError(
                "token_hidden_states count does not match action count: "
                f"{len(per_sample_hidden_states)} != {len(action_token_ids)}"
            )

        pooled_features: list[torch.Tensor] = []
        for sample_hidden_states, state_ids, action_ids in zip(
            per_sample_hidden_states,
            batch_state_token_ids,
            action_token_ids,
        ):
            if not isinstance(sample_hidden_states, torch.Tensor):
                raise TypeError(
                    "each token hidden-state sample must be a torch.Tensor, "
                    f"got {type(sample_hidden_states).__name__}"
                )
            if sample_hidden_states.ndim != 2:
                raise ValueError(
                    "each token hidden-state sample must be 2D, "
                    f"got shape {tuple(sample_hidden_states.shape)}"
                )
            if not action_ids:
                raise ValueError("action_token_ids must be non-empty")

            action_start = len(state_ids)
            action_end = action_start + len(action_ids)
            if action_end > sample_hidden_states.size(0):
                raise ValueError(
                    "action span exceeds token_hidden_states length: "
                    f"{action_end} > {sample_hidden_states.size(0)}"
                )

            action_hidden = sample_hidden_states[action_start:action_end]
            pooled_features.append(
                self._pool_action_span(
                    action_hidden=action_hidden,
                    pooling=pooling,
                    last_k=last_k,
                )
            )

        return torch.stack(pooled_features, dim=0).detach().float()

    def _pool_action_span(
        self,
        *,
        action_hidden: torch.Tensor,
        pooling: str,
        last_k: int,
    ) -> torch.Tensor:
        if action_hidden.ndim != 2:
            raise ValueError(f"action_hidden must be 2D, got shape {tuple(action_hidden.shape)}")
        if action_hidden.size(0) == 0:
            raise ValueError("action_hidden must contain at least one action token")

        if pooling == "last":
            return action_hidden[-1]
        if pooling == "action_mean":
            return action_hidden.mean(dim=0)
        if pooling == "last_k_mean":
            return action_hidden[-min(last_k, action_hidden.size(0)) :].mean(dim=0)

        raise ValueError(
            "hidden_state.token_pooling must be one of "
            "'last', 'action_mean', or 'last_k_mean'; "
            f"got {pooling!r}"
        )

    def _normalize_state_token_ids(
        self,
        state_token_ids: Sequence[int] | Sequence[Sequence[int]],
        batch_size: int,
    ) -> list[list[int]]:
        if batch_size == 0:
            return []
        if not state_token_ids:
            return [[] for _ in range(batch_size)]

        first_item = state_token_ids[0]  # type: ignore[index]
        if isinstance(first_item, int):
            broadcast = list(state_token_ids)  # type: ignore[arg-type]
            return [broadcast[:] for _ in range(batch_size)]

        normalized = [list(item) for item in state_token_ids]  # type: ignore[arg-type]
        if len(normalized) != batch_size:
            raise ValueError(
                f"state_token_ids batch size mismatch: expected {batch_size}, got {len(normalized)}"
            )
        return normalized

    def _select_hidden_states(self, hidden_states: Any) -> Any:
        """按配置选择目标层；如果上游已传入单层 tensor，则直接使用。"""
        if torch is None:
            raise ModuleNotFoundError("torch is required to select hidden states")
        if isinstance(hidden_states, (list, tuple)):
            if not hidden_states:
                raise ValueError("hidden_states sequence is empty")
            layer_index = int(getattr(self.config.hidden_state, "layer", -1))
            try:
                selected = hidden_states[layer_index]
            except IndexError as exc:
                raise IndexError(
                    f"hidden_state.layer={layer_index} is out of range for "
                    f"{len(hidden_states)} hidden-state tensors"
                ) from exc
            if not isinstance(selected, torch.Tensor):
                raise TypeError(
                    "selected hidden state layer must be a torch.Tensor, "
                    f"got {type(selected).__name__}"
                )
            return selected
        return hidden_states
