from __future__ import annotations

"""Q-critic 接口。"""

from dataclasses import dataclass, field
from typing import Any, Sequence

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from mclaw.config import QCriticConfig
from mclaw.core.tree_node import CriticSample

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

    def freeze_backbone(self) -> None:
        """冻结共享 backbone，仅保留 Q-head 可训练。"""
        raise NotImplementedError("TODO: 实现 backbone 冻结逻辑。")

    def score_actions(
        self,
        state_token_ids: Sequence[int] | Sequence[Sequence[int]],
        action_token_ids: Sequence[Sequence[int]],
    ) -> QCriticOutput:
        """对一批 state-action 对做前向打分。"""
        raise NotImplementedError("TODO: 实现 Q-critic 前向打分逻辑。")

    def estimate_state_value(self, q_values: torch.Tensor) -> torch.Tensor:
        """由当前候选动作的 Q 值估计 V(s)。"""
        raise NotImplementedError("TODO: 实现状态值估计逻辑。")

    def build_td_targets(self, samples: Sequence[CriticSample]) -> torch.Tensor:
        """根据 executed transitions 构造 TD target。"""
        raise NotImplementedError("TODO: 实现 TD target 构造逻辑。")

    def update(self, critic_data: Any) -> dict[str, float]:
        """执行一次 Q-head 更新，并返回指标。"""
        raise NotImplementedError("TODO: 实现 Q-head 更新逻辑。")
