from __future__ import annotations

"""Step-level advantage 接口。"""

from dataclasses import dataclass, field
import logging
import math
from typing import Any, Sequence

from mclaw.core.contracts import AuxiliarySample
from mclaw.core.tree_node import TreeNode

logger = logging.getLogger(__name__)


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
    if gamma < 0.0 or gamma > 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")

    updated_nodes: list[TreeNode] = []

    for root in roots:
        root.state_value = estimate_state_value(root.children)

    executed_nodes = sorted(
        _iter_executed_nodes(roots),
        key=lambda node: (node.depth, node.node_id or ""),
        reverse=True,
    )
    parent_state_values = _cache_parent_state_values(executed_nodes)

    for node in executed_nodes:
        current_state_value = parent_state_values[_parent_state_value_key(node)]
        if node.done:
            next_state_value = 0.0
        elif not node.children:
            logger.warning(
                "Node %s at depth %d is not done but has no children; using V(s')=0.0",
                node.node_id,
                node.depth,
            )
            next_state_value = 0.0
        else:
            next_state_value = estimate_state_value(node.children)

        node.state_value = current_state_value
        node.td_target = float(node.env_reward) + (0.0 if node.done else gamma * next_state_value)
        # Plan 约定：被执行节点的真实 TD target 覆盖 Q-head 估计。
        node.q_value = node.td_target
        node.advantage = node.td_target - current_state_value
        updated_nodes.append(node)

    return AdvantageComputationResult(
        updated_nodes=sorted(updated_nodes, key=lambda item: (item.depth, item.node_id or "")),
        metadata={
            "gamma": gamma,
            "n_roots": len(roots),
            "n_executed_nodes": len(updated_nodes),
            "n_auxiliary_samples": 0,
        },
    )


def estimate_state_value(children: Sequence[TreeNode]) -> float:
    """根据同一状态下的候选动作估计 V(s)。"""
    valid_q_values = [
        float(node.q_value)
        for node in children
        if node.q_value is not None and _is_finite_scalar(node.q_value)
    ]
    if valid_q_values:
        return sum(valid_q_values) / len(valid_q_values)

    valid_td_targets = [
        float(node.td_target)
        for node in children
        if node.executed and node.td_target is not None and _is_finite_scalar(node.td_target)
    ]
    if valid_td_targets:
        return sum(valid_td_targets) / len(valid_td_targets)

    logger.warning(
        "estimate_state_value fell back to 0.0: no valid q_value or td_target found "
        "for %d child nodes",
        len(children),
    )
    return 0.0


def _iter_executed_nodes(roots: Sequence[TreeNode]) -> list[TreeNode]:
    executed_nodes: list[TreeNode] = []
    stack = list(reversed(list(roots)))
    while stack:
        node = stack.pop()
        stack.extend(reversed(node.children))
        if node.executed:
            executed_nodes.append(node)
    return executed_nodes


def _cache_parent_state_values(executed_nodes: Sequence[TreeNode]) -> dict[int, float]:
    parent_state_values: dict[int, float] = {}
    for node in executed_nodes:
        parent_key = _parent_state_value_key(node)
        if parent_key in parent_state_values:
            continue
        current_children = node.parent.children if node.parent is not None else [node]
        parent_state_values[parent_key] = estimate_state_value(current_children)
    return parent_state_values


def _parent_state_value_key(node: TreeNode) -> int:
    return id(node.parent) if node.parent is not None else id(node)


def _is_finite_scalar(value: Any) -> bool:
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(scalar)
