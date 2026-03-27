from __future__ import annotations

"""树结构和训练样本的数据接口。"""

from dataclasses import dataclass, field
from typing import Any

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclass(slots=True)
class EnvironmentStep:
    """单步环境反馈。"""

    reward: float = 0.0
    next_state: Any | None = None
    done: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TreeNode:
    """树中的一个状态-动作节点。"""

    state_tokens: list[int]
    action_tokens: list[int] = field(default_factory=list)
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


@dataclass(slots=True)
class AuxiliarySample:
    """同簇 sibling 的辅助监督样本。"""

    state_tokens: list[int]
    action_tokens: list[int]
    advantage: float | None = None
    td_target: float | None = None
    cluster_id: int = -1
    cluster_weight: float = 1.0
    source_node_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CriticSample:
    """Q-head TD 更新使用的样本。"""

    state_tokens: list[int]
    action_tokens: list[int]
    reward: float
    next_state_tokens: list[int] = field(default_factory=list)
    done: bool = False
    gamma: float = 0.99
    source_node_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TreeRolloutOutput:
    """一次树状 rollout 的统一输出。"""

    actor_data: Any = None
    aux_actor_data: Any = None
    critic_data: Any = None
    roots: list[TreeNode] = field(default_factory=list)
    aux_samples: list[AuxiliarySample] = field(default_factory=list)
    critic_samples: list[CriticSample] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
