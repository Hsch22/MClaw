from __future__ import annotations

"""候选代表选择接口。"""

from dataclasses import dataclass, field
from typing import Any, Sequence

from .tree_node import AuxiliarySample, TreeNode


@dataclass(slots=True)
class SelectionResult:
    """一次候选选择的结果。"""

    selected_index: int
    representative_indices: list[int] = field(default_factory=list)
    cluster_labels: list[int] = field(default_factory=list)
    auxiliary_indices: list[int] = field(default_factory=list)
    cluster_weights: dict[int, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class BranchSelector:
    """负责根节点代表选择和分支内动作选择。"""

    def __init__(self, weighting_mode: str = "equal_per_selected_cluster") -> None:
        self.weighting_mode = weighting_mode

    def select_root_representatives(
        self,
        candidates: Sequence[TreeNode],
        representative_indices: Sequence[int],
        cluster_labels: Sequence[int],
        n_select: int,
    ) -> list[int]:
        """从根节点代表集合中挑出要执行的代表动作。"""
        raise NotImplementedError("TODO: 实现根节点代表选择逻辑。")

    def select_branch_action(
        self,
        candidates: Sequence[TreeNode],
        representative_indices: Sequence[int],
        cluster_labels: Sequence[int],
    ) -> SelectionResult:
        """在单个分支内选出唯一执行动作，并标记辅助样本。"""
        raise NotImplementedError("TODO: 实现分支内动作选择逻辑。")

    def build_auxiliary_samples(
        self,
        parent: TreeNode,
        candidates: Sequence[TreeNode],
        selection: SelectionResult,
        advantage: float,
        td_target: float | None = None,
    ) -> list[AuxiliarySample]:
        """把被选簇中的未执行 siblings 转成 auxiliary samples。"""
        raise NotImplementedError("TODO: 实现 auxiliary sample 构造逻辑。")

    def compute_cluster_weights(
        self,
        cluster_labels: Sequence[int],
        selected_indices: Sequence[int],
    ) -> dict[int, float]:
        """按簇等权策略生成辅助监督权重。"""
        raise NotImplementedError("TODO: 实现簇等权权重计算逻辑。")
