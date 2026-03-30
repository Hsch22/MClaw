from __future__ import annotations

"""候选代表选择接口。"""

from dataclasses import dataclass, field
import math
from typing import Any, Sequence

from .contracts import AuxiliarySample
from .tree_node import TreeNode, resolve_action_token_weight


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
        """从根节点代表集合中挑出要执行的代表动作。如果给定k个分支，m条rollout。先对rollout进行聚类，排序，在前k个簇中，每个簇挑出1个代表动作，执行。"""
        if n_select <= 0:
            return []

        normalized_representatives = self._normalize_representatives(
            candidates=candidates,
            representative_indices=representative_indices,
            cluster_labels=cluster_labels,
        )
        print(
            "[MClaw][BranchSelector] 根节点代表排序方法: "
            f"{self._describe_ranking_method(candidates, normalized_representatives)}"
        )
        ranked_representatives = self._rank_indices(candidates, normalized_representatives)
        selected_indices = ranked_representatives[: min(n_select, len(ranked_representatives))]

        selected_index_set = set(selected_indices)
        for index, node in enumerate(candidates):
            node.selected_for_execution = index in selected_index_set

        return selected_indices

    def select_branch_action(
        self,
        candidates: Sequence[TreeNode],
        representative_indices: Sequence[int],
        cluster_labels: Sequence[int],
    ) -> SelectionResult:
        """在单个分支内选出唯一执行动作，并标记同一个簇中的其他样本为辅助样本。"""
        normalized_representatives = self._normalize_representatives(
            candidates=candidates,
            representative_indices=representative_indices,
            cluster_labels=cluster_labels,
        )
        ranked_representatives = self._rank_indices(candidates, normalized_representatives)
        selected_index = ranked_representatives[0]
        selected_cluster_id = cluster_labels[selected_index]
        auxiliary_indices = [
            index
            for index, cluster_id in enumerate(cluster_labels)
            if cluster_id == selected_cluster_id and index != selected_index
        ]
        cluster_weights = self.compute_cluster_weights(
            cluster_labels=cluster_labels,
            selected_indices=[selected_index],
        )

        for index, node in enumerate(candidates):
            node.selected_for_execution = index == selected_index
            node.selected_for_aux_loss = index in auxiliary_indices

        return SelectionResult(
            selected_index=selected_index,
            representative_indices=ranked_representatives,
            cluster_labels=list(cluster_labels),
            auxiliary_indices=auxiliary_indices,
            cluster_weights=cluster_weights,
            metadata={
                "selected_cluster_id": selected_cluster_id,
                "selected_score": self._candidate_score(candidates[selected_index]),
                "n_auxiliary": len(auxiliary_indices),
            },
        )

    def build_auxiliary_samples(
        self,
        parent: TreeNode,
        candidates: Sequence[TreeNode],
        selection: SelectionResult,
        advantage: float,
        td_target: float | None = None,
    ) -> list[AuxiliarySample]:
        """把被选簇中的未执行 siblings 转成 auxiliary samples。"""
        self._validate_selection(candidates, selection)

        selected_node = candidates[selection.selected_index]
        auxiliary_samples: list[AuxiliarySample] = []

        for candidate_index in selection.auxiliary_indices:
            node = candidates[candidate_index]
            cluster_id = self._resolve_cluster_id(
                node=node,
                candidate_index=candidate_index,
                cluster_labels=selection.cluster_labels,
            )
            auxiliary_samples.append(
                AuxiliarySample(
                    state_tokens=list(node.state_tokens or parent.state_tokens),
                    action_tokens=list(node.action_tokens),
                    advantage=advantage,
                    td_target=td_target,
                    cluster_id=cluster_id,
                    cluster_weight=selection.cluster_weights.get(cluster_id, 1.0),
                    token_weight=resolve_action_token_weight(node.action_tokens),
                    source_node_id=node.node_id,
                    metadata={
                        "parent_node_id": parent.node_id,
                        "selected_node_id": selected_node.node_id,
                        "selected_index": selection.selected_index,
                        "candidate_index": candidate_index,
                    },
                )
            )

        return auxiliary_samples

    def compute_cluster_weights(
        self,
        cluster_labels: Sequence[int],
        selected_indices: Sequence[int],
    ) -> dict[int, float]:
        """按簇等权策略生成辅助监督权重。"""
        if self.weighting_mode != "equal_per_selected_cluster":
            raise ValueError(f"Unsupported weighting mode: {self.weighting_mode}")
        if not cluster_labels or not selected_indices:
            return {}

        self._validate_cluster_labels(cluster_labels)

        n_candidates = len(cluster_labels)
        selected_index_set = set(selected_indices)
        if any(index < 0 or index >= n_candidates for index in selected_index_set):
            raise IndexError("selected_indices contains an out-of-range index")

        selected_cluster_ids: list[int] = []
        seen_cluster_ids: set[int] = set()
        for index in selected_indices:
            cluster_id = cluster_labels[index]
            if cluster_id in seen_cluster_ids:
                continue
            seen_cluster_ids.add(cluster_id)
            selected_cluster_ids.append(cluster_id)

        if not selected_cluster_ids:
            return {}

        cluster_weights: dict[int, float] = {}
        n_selected_clusters = len(selected_cluster_ids)
        for cluster_id in selected_cluster_ids:
            auxiliary_count = sum(
                1
                for index, label in enumerate(cluster_labels)
                if label == cluster_id and index not in selected_index_set
            )
            if auxiliary_count <= 0:
                continue
            cluster_weights[cluster_id] = 1.0 / n_selected_clusters / auxiliary_count

        return cluster_weights

    def _normalize_representatives(
        self,
        candidates: Sequence[TreeNode],
        representative_indices: Sequence[int],
        cluster_labels: Sequence[int],
    ) -> list[int]:
        if not candidates:
            raise ValueError("candidates must be non-empty")
        if not representative_indices:
            raise ValueError("representative_indices must be non-empty")
        if len(cluster_labels) != len(candidates):
            raise ValueError("cluster_labels must have the same length as candidates")

        self._validate_cluster_labels(cluster_labels)

        normalized_indices: list[int] = []
        seen_cluster_ids: set[int] = set()
        for index in representative_indices:
            if index < 0 or index >= len(candidates):
                raise IndexError("representative_indices contains an out-of-range index")
            cluster_id = cluster_labels[index]
            if cluster_id in seen_cluster_ids:
                continue
            seen_cluster_ids.add(cluster_id)
            normalized_indices.append(index)

        if not normalized_indices:
            raise ValueError("representative_indices does not contain a valid representative")

        return normalized_indices

    def _validate_cluster_labels(self, cluster_labels: Sequence[int]) -> None:
        if any(cluster_id < 0 for cluster_id in cluster_labels):
            raise ValueError("cluster_labels must be assigned before selection")

    def _rank_indices(self, candidates: Sequence[TreeNode], indices: Sequence[int]) -> list[int]:
        scored_indices = [
            (self._candidate_score(candidates[index]), order, index)
            for order, index in enumerate(indices)
        ]
        scored_indices.sort(key=lambda item: (-item[0], item[1]))
        return [index for _, _, index in scored_indices]

    def _candidate_score(self, node: TreeNode) -> float:
        for value in (node.q_value, node.log_prob):
            if value is None:
                continue
            try:
                score = float(value)
            except (TypeError, ValueError):
                continue
            if math.isnan(score):
                continue
            return score
        return float("-inf")

    def _describe_ranking_method(
        self,
        candidates: Sequence[TreeNode],
        representative_indices: Sequence[int],
    ) -> str:
        if any(candidates[index].q_value is not None for index in representative_indices):
            return "按 representative 的 q_value 降序；若分数并列则保持 representative_indices 原顺序"
        if any(candidates[index].log_prob is not None for index in representative_indices):
            return "按 representative 的 log_prob 降序；若分数并列则保持 representative_indices 原顺序"
        return "未提供 q_value / log_prob，直接保持 representative_indices 原顺序"

    def _validate_selection(
        self,
        candidates: Sequence[TreeNode],
        selection: SelectionResult,
    ) -> None:
        if not candidates:
            raise ValueError("candidates must be non-empty")
        if selection.selected_index < 0 or selection.selected_index >= len(candidates):
            raise IndexError("selection.selected_index is out of range")
        if any(index < 0 or index >= len(candidates) for index in selection.auxiliary_indices):
            raise IndexError("selection.auxiliary_indices contains an out-of-range index")

    def _resolve_cluster_id(
        self,
        node: TreeNode,
        candidate_index: int,
        cluster_labels: Sequence[int],
    ) -> int:
        if cluster_labels:
            return cluster_labels[candidate_index]
        if node.cluster_id < 0:
            raise ValueError("candidate cluster_id is not assigned")
        return node.cluster_id
