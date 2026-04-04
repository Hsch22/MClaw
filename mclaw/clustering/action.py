from __future__ import annotations

"""基于原始 action token 序列的聚类器。"""

from typing import Any, Mapping, Sequence

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from mclaw.core.tree_node import TreeNode

from .base import BaseClusterer, ClusterResult


class ActionClusterer(BaseClusterer):
    """按完整 action token 序列做精确分组。"""

    def extract_features(
        self,
        action_token_ids: Sequence[Sequence[int]],
        state_token_ids: Sequence[int] | Sequence[Sequence[int]],
        model_outputs: Mapping[str, Any],
    ) -> torch.Tensor:
        """把变长 action token 序列整理成固定维度特征。"""
        del state_token_ids, model_outputs
        self._require_torch()

        max_length = max((len(action_ids) for action_ids in action_token_ids), default=1)
        feature_length = max(max_length, 1)
        features = torch.zeros(
            (len(action_token_ids), feature_length),
            dtype=torch.float32,
        )
        for row_index, action_ids in enumerate(action_token_ids):
            if not action_ids:
                continue
            # 为了与 padding=0 区分，真实 token id 统一平移到正区间。
            shifted = [float(token_id) + 1.0 for token_id in action_ids]
            features[row_index, : len(shifted)] = torch.tensor(shifted, dtype=torch.float32)
        return features

    def cluster_candidates(
        self,
        nodes: Sequence[TreeNode],
        n_clusters: int,
        model_outputs: Mapping[str, Any],
    ) -> ClusterResult:
        """直接按 action token 序列精确分组，不走 PCA / K-Means。"""
        del model_outputs
        self._require_torch()
        if not nodes:
            raise ValueError("nodes must be non-empty")

        action_token_ids = [list(node.action_tokens) for node in nodes]
        first_state_tokens = list(nodes[0].state_tokens)
        shared_state = all(list(node.state_tokens) == first_state_tokens for node in nodes)
        state_token_ids: Sequence[int] | Sequence[Sequence[int]]
        if shared_state:
            state_token_ids = first_state_tokens
        else:
            state_token_ids = [list(node.state_tokens) for node in nodes]

        features = self.extract_features(
            action_token_ids=action_token_ids,
            state_token_ids=state_token_ids,
            model_outputs={},
        )
        print(
            "[MClaw][Clusterer] 候选动作聚类方法: "
            f"特征={self._feature_method_name()}, "
            "聚类算法=exact_action_token_match, "
            f"目标簇数={n_clusters}, "
            "PCA目标维度=ignored"
        )

        labels: list[int] = []
        representative_indices: list[int] = []
        cluster_id_by_action: dict[tuple[int, ...], int] = {}
        cluster_sizes: list[int] = []
        for candidate_index, action_ids in enumerate(action_token_ids):
            action_key = tuple(int(token_id) for token_id in action_ids)
            cluster_id = cluster_id_by_action.get(action_key)
            if cluster_id is None:
                cluster_id = len(cluster_id_by_action)
                cluster_id_by_action[action_key] = cluster_id
                representative_indices.append(candidate_index)
                cluster_sizes.append(0)
            labels.append(cluster_id)
            cluster_sizes[cluster_id] += 1

        result = ClusterResult(
            labels=labels,
            representative_indices=representative_indices,
            features=features,
            metadata={
                "n_samples": len(nodes),
                "requested_clusters": n_clusters,
                "n_clusters": len(representative_indices),
                "original_dim": int(features.size(1)),
                "cluster_dim": int(features.size(1)),
                "pca_applied": False,
                "kmeans_iterations": 0,
                "shared_state": shared_state,
                "cluster_algorithm": "exact_action_token_match",
                "cluster_sizes": cluster_sizes,
                "requested_clusters_ignored": True,
            },
        )
        print(
            "[MClaw][Clusterer] 聚类完成: "
            f"特征方法={self._feature_method_name()}, "
            f"实际簇数={result.metadata.get('n_clusters')}, "
            "PCA是否启用=False, "
            "KMeans迭代数=0, "
            f"代表索引={result.representative_indices}"
        )

        for index, node in enumerate(nodes):
            node.cluster_id = result.labels[index]
            node.cluster_feature = result.features[index].detach().clone()

        return result
