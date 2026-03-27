from __future__ import annotations

"""Step-level advantage 接口。"""

from dataclasses import dataclass, field
from typing import Any, Sequence

from mclaw.core.tree_node import AuxiliarySample, TreeNode


@dataclass(slots=True)
class AdvantageComputationResult:
    """Advantage 计算结果。"""

    updated_nodes: list[TreeNode] = field(default_factory=list)
    auxiliary_samples: list[AuxiliarySample] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def compute_tree_advantage(
    roots: Sequence[TreeNode],
    gamma: float = 0.99,
) -> AdvantageComputationResult:
    """在树结构上计算 executed nodes 的 step-level advantage。"""
    raise NotImplementedError("TODO: 实现树结构 advantage 计算逻辑。")


def estimate_state_value(children: Sequence[TreeNode]) -> float:
    """根据同一状态下的候选动作估计 V(s)。"""
    raise NotImplementedError("TODO: 实现状态值估计逻辑。")


def propagate_auxiliary_targets(
    nodes: Sequence[TreeNode],
    cluster_id: int,
    advantage: float,
    td_target: float | None = None,
) -> list[AuxiliarySample]:
    """把被选簇的监督信号扩散到未执行 siblings。"""
    raise NotImplementedError("TODO: 实现 auxiliary target 传播逻辑。")
