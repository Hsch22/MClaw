from __future__ import annotations

"""Q-critic 接口。"""

from dataclasses import dataclass, field
import math
from typing import Any, Sequence

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from mclaw.config import QCriticConfig
from mclaw.core.contracts import CriticSample

from .q_head import QHead


@dataclass(slots=True)
class QCriticOutput:
    """Q-critic 前向输出。"""

    hidden_states: torch.Tensor | None = None
    q_values: torch.Tensor | None = None
    action_last_token_indices: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class QCritic:
    """共享 backbone + Q-head 的动作价值估计接口。"""

    def __init__(
        self,
        actor_module_fsdp: Any,
        q_head: QHead,
        tokenizer: Any,
        config: QCriticConfig,
    ) -> None:
        self.actor_module_fsdp = actor_module_fsdp
        self.q_head = q_head
        self.tokenizer = tokenizer
        self.config = config
        self.optimizer = None
        if torch is not None and hasattr(self.q_head, "parameters"):
            self.optimizer = torch.optim.Adam(self.q_head.parameters(), lr=self.config.lr)

    def score_actions(
        self,
        state_token_ids: Sequence[int] | Sequence[Sequence[int]],
        action_token_ids: Sequence[Sequence[int]],
    ) -> QCriticOutput:
        """对一批 state-action 对做前向打分。"""
        if torch is None:
            raise ModuleNotFoundError("torch is required to score actions")
        hidden_states, action_last_token_indices, metadata = self._encode_state_action_pairs(
            state_token_ids=state_token_ids,
            action_token_ids=action_token_ids,
        )
        with torch.no_grad():
            q_values = self.q_head(hidden_states)
        return QCriticOutput(
            hidden_states=hidden_states,
            q_values=q_values,
            action_last_token_indices=action_last_token_indices,
            metadata=metadata,
        )

    def estimate_state_value(self, q_values: torch.Tensor) -> torch.Tensor:
        """由当前候选动作的 Q 值估计 V(s)。"""
        if torch is None:
            raise ModuleNotFoundError("torch is required to estimate state value")
        if not isinstance(q_values, torch.Tensor):
            raise TypeError("q_values must be a torch.Tensor")
        if q_values.ndim == 0:
            return q_values
        if q_values.numel() == 0:
            return torch.tensor(0.0, dtype=torch.float32, device=q_values.device)
        sanitized = torch.nan_to_num(q_values.float(), nan=0.0, posinf=0.0, neginf=0.0)
        return sanitized.mean(dim=-1)

    def build_td_targets(self, samples: Sequence[CriticSample]) -> torch.Tensor:
        """根据 executed transitions 构造 TD target。"""
        if torch is None:
            raise ModuleNotFoundError("torch is required to build TD targets")

        gamma = float(getattr(self.config, "gamma", 0.99))
        targets: list[float] = []
        for sample in samples:
            next_state_value = 0.0
            if not sample.done:
                next_state_value = _require_next_state_value(sample)
            targets.append(float(sample.reward) + (0.0 if sample.done else gamma * next_state_value))
        return torch.tensor(targets, dtype=torch.float32, device=self._resolve_device())

    def update(self, critic_data: Any) -> dict[str, float]:
        """执行一次 Q-head 更新，并返回指标。"""
        if torch is None:
            raise ModuleNotFoundError("torch is required to update QCritic")

        samples = critic_data.samples if hasattr(critic_data, "samples") else critic_data
        if not samples:
            return {
                "critic/q_loss": 0.0,
                "critic/q_mean": 0.0,
                "critic/target_mean": 0.0,
            }

        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.q_head.parameters(), lr=self.config.lr)

        self.q_head.train()
        hidden_states, _, _ = self._encode_state_action_pairs(
            state_token_ids=[sample.state_tokens for sample in samples],
            action_token_ids=[sample.action_tokens for sample in samples],
        )
        q_values = self.q_head(hidden_states.detach())
        targets = self.build_td_targets(samples).to(q_values.device)

        loss = torch.nn.functional.mse_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        grad_clip_norm = getattr(self.config, "grad_clip_norm", None)
        if grad_clip_norm is not None and grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(self.q_head.parameters(), max_norm=grad_clip_norm)
        self.optimizer.step()

        return {
            "critic/q_loss": float(loss.detach().item()),
            "critic/q_mean": float(q_values.detach().mean().item()),
            "critic/target_mean": float(targets.detach().mean().item()),
        }

    def _encode_state_action_pairs(
        self,
        state_token_ids: Sequence[int] | Sequence[Sequence[int]],
        action_token_ids: Sequence[Sequence[int]],
    ) -> tuple[torch.Tensor, list[int], dict[str, Any]]:
        if torch is None:
            raise ModuleNotFoundError("torch is required to run QCritic")
        if self.actor_module_fsdp is None:
            raise ValueError("actor_module_fsdp is required to run QCritic")

        batch_state_token_ids = _normalize_state_token_ids(state_token_ids, len(action_token_ids))
        if len(batch_state_token_ids) != len(action_token_ids):
            raise ValueError("state_token_ids and action_token_ids must have the same batch size")

        sequences: list[list[int]] = []
        action_last_token_indices: list[int] = []
        for state_ids, action_ids in zip(batch_state_token_ids, action_token_ids):
            if not action_ids:
                raise ValueError("action_token_ids must be non-empty")
            merged = list(state_ids) + list(action_ids)
            sequences.append(merged)
            action_last_token_indices.append(len(merged) - 1)

        input_ids, attention_mask = _pad_sequences(
            sequences=sequences,
            pad_token_id=_resolve_pad_token_id(self.tokenizer),
            device=self._resolve_device(),
        )

        backbone_was_training = None
        if hasattr(self.actor_module_fsdp, "training"):
            backbone_was_training = bool(self.actor_module_fsdp.training)
            self.actor_module_fsdp.eval()

        try:
            with torch.no_grad():
                model_outputs = self.actor_module_fsdp(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                last_hidden_state = _resolve_last_hidden_state(model_outputs)
                hidden_states = last_hidden_state[
                    torch.arange(last_hidden_state.size(0), device=last_hidden_state.device),
                    torch.tensor(action_last_token_indices, device=last_hidden_state.device),
                ]
        finally:
            if backbone_was_training:
                self.actor_module_fsdp.train()

        metadata: dict[str, Any] = {
            "attention_mask": attention_mask.detach(),
            "sequence_lengths": [len(sequence) for sequence in sequences],
        }
        logits = getattr(model_outputs, "logits", None)
        if logits is not None:
            metadata["logits"] = logits.detach()

        return hidden_states, action_last_token_indices, metadata

    def _resolve_device(self) -> torch.device:
        if torch is None:
            raise ModuleNotFoundError("torch is required to resolve device")
        for module in (self.q_head, self.actor_module_fsdp):
            if module is None:
                continue
            try:
                parameter = next(module.parameters())
            except (StopIteration, AttributeError, TypeError):
                continue
            return parameter.device
        return torch.device("cpu")


def _normalize_state_token_ids(
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


def _pad_sequences(
    sequences: Sequence[Sequence[int]],
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_length = max((len(sequence) for sequence in sequences), default=0)
    input_ids = torch.full(
        (len(sequences), max_length),
        fill_value=pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros(
        (len(sequences), max_length),
        dtype=torch.long,
        device=device,
    )
    for row_index, sequence in enumerate(sequences):
        if not sequence:
            continue
        seq_tensor = torch.tensor(sequence, dtype=torch.long, device=device)
        input_ids[row_index, : len(sequence)] = seq_tensor
        attention_mask[row_index, : len(sequence)] = 1
    return input_ids, attention_mask


def _resolve_pad_token_id(tokenizer: Any) -> int:
    for attribute in ("pad_token_id", "eos_token_id", "bos_token_id"):
        value = getattr(tokenizer, attribute, None)
        if isinstance(value, int) and value >= 0:
            return value
    return 0


def _resolve_last_hidden_state(model_outputs: Any) -> torch.Tensor:
    hidden_states = getattr(model_outputs, "hidden_states", None)
    if hidden_states is not None:
        if isinstance(hidden_states, (list, tuple)):
            if not hidden_states:
                raise AttributeError("model_outputs.hidden_states is empty")
            return hidden_states[-1]
        return hidden_states
    last_hidden_state = getattr(model_outputs, "last_hidden_state", None)
    if last_hidden_state is None:
        raise AttributeError("model_outputs must expose hidden_states or last_hidden_state")
    return last_hidden_state


def _require_next_state_value(sample: CriticSample) -> float:
    metadata = sample.metadata
    if metadata is None or "next_state_value" not in metadata:
        raise KeyError(
            "critic sample is missing required metadata['next_state_value'] "
            f"for source_node_id={sample.source_node_id!r}"
        )

    value = metadata["next_state_value"]
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "critic sample metadata['next_state_value'] must be a finite scalar, "
            f"got {value!r} for source_node_id={sample.source_node_id!r}"
        ) from exc
    if not math.isfinite(scalar):
        raise ValueError(
            "critic sample metadata['next_state_value'] must be finite, "
            f"got {scalar!r} for source_node_id={sample.source_node_id!r}"
        )
    return scalar
