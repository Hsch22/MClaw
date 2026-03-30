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
        del action_token_ids, state_token_ids
        self._require_torch()

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
