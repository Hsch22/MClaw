from __future__ import annotations

"""聚类器基类接口。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence
import torch
from mclaw.config import ClusteringConfig
from mclaw.core.tree_node import TreeNode


@dataclass(slots=True)
class ClusterResult:
    """聚类输出。"""

    labels: list[int]
    representative_indices: list[int]
    features: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseClusterer(ABC):
    """所有聚类方案的公共接口。"""

    def __init__(self, config: ClusteringConfig) -> None:
        self.config = config

    @abstractmethod
    def extract_features(
        self,
        action_token_ids: Sequence[Sequence[int]],
        state_token_ids: Sequence[int] | Sequence[Sequence[int]],
        model_outputs: Mapping[str, Any],
    ) -> torch.Tensor:
        """提取候选动作的聚类特征。"""

    def cluster(self, features: torch.Tensor, n_clusters: int) -> ClusterResult:
        """对特征做聚类并返回代表动作。"""
        raise NotImplementedError("TODO: 实现默认聚类流程。")

    def cluster_candidates(
        self,
        nodes: Sequence[TreeNode],
        n_clusters: int,
        model_outputs: Mapping[str, Any],
    ) -> ClusterResult:
        """直接对 TreeNode 列表做特征提取和聚类。"""
        raise NotImplementedError("TODO: 实现节点级聚类流程。")
