from __future__ import annotations

"""树结构数据接口。"""

from dataclasses import dataclass, field
from typing import Any, Sequence

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclass(slots=True)
class TreeNode:
    """树中的一个状态-动作节点。"""

    state_tokens: list[int]
    action_tokens: list[int] = field(default_factory=list)
    next_state_tokens: list[int] = field(default_factory=list)
    parent: TreeNode | None = None
    children: list[TreeNode] = field(default_factory=list)
    depth: int = 0
    state_text: str = ""
    action_text: str = ""
    executed: bool = False
    env_reward: float = 0.0
    env_next_state: Any | None = None
    done: bool = False
    log_prob: float | None = None
    q_value: float | None = None
    state_value: float | None = None
    td_target: float | None = None
    advantage: float | None = None
    cluster_id: int = -1
    cluster_feature: torch.Tensor | None = None
    selected_for_execution: bool = False
    selected_for_aux_loss: bool = False
    branch_id: int | None = None
    node_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def resolve_action_token_weight(action_tokens: Sequence[int]) -> float:
    """按 action token 长度给 step/sample 分配均匀权重。"""
    if not action_tokens:
        return 0.0
    return 1.0 / len(action_tokens)
