from __future__ import annotations

"""聚类器基类接口。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

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
        self._require_torch()

        x = self._prepare_features(features)
        n_samples = x.size(0)
        requested_clusters = n_clusters
        n_clusters = min(max(1, n_clusters), n_samples)

        x_for_cluster, pca_applied = self._maybe_reduce_dim(x)
        labels, centroids, n_iterations = self._run_kmeans(x_for_cluster, n_clusters)
        representative_indices = self._select_representatives(
            x_for_cluster,
            labels,
            centroids,
        )

        return ClusterResult(
            labels=labels.tolist(),
            representative_indices=representative_indices,
            features=x_for_cluster,
            metadata={
                "n_samples": n_samples,
                "requested_clusters": requested_clusters,
                "n_clusters": n_clusters,
                "original_dim": int(x.size(1)),
                "cluster_dim": int(x_for_cluster.size(1)),
                "pca_applied": pca_applied,
                "kmeans_iterations": n_iterations,
            },
        )

    def cluster_candidates(
        self,
        nodes: Sequence[TreeNode],
        n_clusters: int,
        model_outputs: Mapping[str, Any],
    ) -> ClusterResult:
        """直接对 TreeNode 列表做特征提取和聚类。"""
        self._require_torch()
        if not nodes:
            raise ValueError("nodes must be non-empty")

        action_token_ids = [list(node.action_tokens) for node in nodes]
        first_state_tokens = list(nodes[0].state_tokens)
        shared_state = all(list(node.state_tokens) == first_state_tokens for node in nodes)
        if shared_state:
            state_token_ids: Sequence[int] | Sequence[Sequence[int]] = first_state_tokens
        else:
            state_token_ids = [list(node.state_tokens) for node in nodes]

        print(
            "[MClaw][Clusterer] 候选动作聚类方法: "
            f"特征={self._feature_method_name()}, "
            f"聚类算法=deterministic_torch_kmeans, "
            f"目标簇数={n_clusters}, "
            f"PCA目标维度={getattr(self.config, 'pca_dim', 0)}"
        )
        features = self.extract_features(
            action_token_ids=action_token_ids,
            state_token_ids=state_token_ids,
            model_outputs=model_outputs,
        )
        result = self.cluster(features, n_clusters)
        result.metadata["shared_state"] = shared_state
        print(
            "[MClaw][Clusterer] 聚类完成: "
            f"特征方法={self._feature_method_name()}, "
            f"实际簇数={result.metadata.get('n_clusters')}, "
            f"PCA是否启用={result.metadata.get('pca_applied')}, "
            f"KMeans迭代数={result.metadata.get('kmeans_iterations')}, "
            f"代表索引={result.representative_indices}"
        )

        for index, node in enumerate(nodes):
            node.cluster_id = result.labels[index]
            if result.features is not None:
                node.cluster_feature = result.features[index].detach().clone()

        return result

    def _require_torch(self) -> None:
        """在运行期检查 torch 是否可用。"""
        if torch is None:
            raise ModuleNotFoundError("torch is required to use clustering.")

    def _prepare_features(self, features: torch.Tensor) -> torch.Tensor:
        """清洗输入特征，并统一成 2D float tensor。"""
        if not isinstance(features, torch.Tensor):
            raise TypeError("features must be a torch.Tensor")
        if features.ndim != 2:
            raise ValueError(f"features must be 2D, got shape {tuple(features.shape)}")
        if features.size(0) == 0:
            raise ValueError("features must contain at least one sample")
        if features.size(1) == 0:
            raise ValueError("features must contain at least one feature dimension")

        x = features.detach().float().contiguous()
        return torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

    def _maybe_reduce_dim(self, features: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """按配置可选地执行 PCA 降维。"""
        target_dim = int(getattr(self.config, "pca_dim", 0))
        if target_dim <= 0 or features.size(1) <= target_dim or features.size(0) <= 1:
            return features, False

        reduced_dim = min(target_dim, features.size(0), features.size(1))
        if reduced_dim >= features.size(1):
            return features, False

        centered = features - features.mean(dim=0, keepdim=True)
        try:
            _, _, v = torch.pca_lowrank(centered, q=reduced_dim, center=False)
        except RuntimeError:
            return features, False
        return centered @ v[:, :reduced_dim], True

    def _run_kmeans(
        self,
        features: torch.Tensor,
        n_clusters: int,
        max_iterations: int = 30,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """使用一个小型 torch KMeans 默认实现。"""
        if n_clusters < 1:
            raise ValueError("n_clusters must be positive")

        centroids = self._initialize_centroids(features, n_clusters)
        previous_labels: torch.Tensor | None = None

        for iteration in range(max_iterations):
            distances = torch.cdist(features, centroids)
            labels = distances.argmin(dim=1)
            labels = self._ensure_non_empty_clusters(labels, distances, n_clusters)

            new_centroids = []
            for cluster_id in range(n_clusters):
                members = features[labels == cluster_id]
                new_centroids.append(members.mean(dim=0))
            centroids = torch.stack(new_centroids, dim=0)

            if previous_labels is not None and torch.equal(labels, previous_labels):
                return labels, centroids, iteration + 1
            previous_labels = labels

        final_distances = torch.cdist(features, centroids)
        final_labels = self._ensure_non_empty_clusters(
            final_distances.argmin(dim=1),
            final_distances,
            n_clusters,
        )
        final_centroids = []
        for cluster_id in range(n_clusters):
            members = features[final_labels == cluster_id]
            final_centroids.append(members.mean(dim=0))
        return final_labels, torch.stack(final_centroids, dim=0), max_iterations

    def _initialize_centroids(self, features: torch.Tensor, n_clusters: int) -> torch.Tensor:
        """用确定性的最远点采样初始化质心。"""
        mean_point = features.mean(dim=0, keepdim=True)
        first_index = torch.norm(features - mean_point, dim=1).argmax().item()
        chosen_indices = [first_index]

        while len(chosen_indices) < n_clusters:
            chosen = features[chosen_indices]
            distances = torch.cdist(features, chosen)
            min_distances = distances.min(dim=1).values
            for index in chosen_indices:
                min_distances[index] = -1.0
            next_index = min_distances.argmax().item()
            if next_index in chosen_indices:
                break
            chosen_indices.append(next_index)

        if len(chosen_indices) < n_clusters:
            for index in range(features.size(0)):
                if index not in chosen_indices:
                    chosen_indices.append(index)
                if len(chosen_indices) == n_clusters:
                    break

        return features[torch.tensor(chosen_indices, device=features.device)]

    def _ensure_non_empty_clusters(
        self,
        labels: torch.Tensor,
        distances: torch.Tensor,
        n_clusters: int,
    ) -> torch.Tensor:
        """通过重分配远离质心的样本，必要时用最远点重置空簇。"""
        adjusted = labels.clone()
        counts = torch.bincount(adjusted, minlength=n_clusters)
        empty_clusters = (counts == 0).nonzero(as_tuple=False).flatten().tolist()
        if not empty_clusters:
            return adjusted

        min_distances = distances.min(dim=1).values
        donor_indices = self._collect_donor_indices(
            labels=adjusted,
            min_distances=min_distances,
            n_clusters=n_clusters,
        )

        if len(donor_indices) >= len(empty_clusters):
            for empty_cluster, sample_index in zip(empty_clusters, donor_indices):
                donor_cluster = int(adjusted[sample_index].item())
                adjusted[sample_index] = empty_cluster
                counts[donor_cluster] -= 1
                counts[empty_cluster] += 1
            return adjusted

        adjusted = self._reseed_empty_clusters(
            labels=adjusted,
            min_distances=min_distances,
            n_clusters=n_clusters,
        )
        counts = torch.bincount(adjusted, minlength=n_clusters)
        if (counts == 0).any():
            raise RuntimeError("failed to reseed empty clusters")

        return adjusted

    def _collect_donor_indices(
        self,
        labels: torch.Tensor,
        min_distances: torch.Tensor,
        n_clusters: int,
    ) -> list[int]:
        donor_indices: list[int] = []
        for cluster_id in range(n_clusters):
            member_indices = (labels == cluster_id).nonzero(as_tuple=False).flatten().tolist()
            if len(member_indices) <= 1:
                continue
            member_indices.sort(key=lambda index: float(min_distances[index]), reverse=True)
            donor_indices.extend(member_indices[:-1])

        donor_indices.sort(key=lambda index: float(min_distances[index]), reverse=True)
        return donor_indices

    def _reseed_empty_clusters(
        self,
        labels: torch.Tensor,
        min_distances: torch.Tensor,
        n_clusters: int,
    ) -> torch.Tensor:
        adjusted = labels.clone()
        counts = torch.bincount(adjusted, minlength=n_clusters)
        empty_clusters = (counts == 0).nonzero(as_tuple=False).flatten().tolist()
        if not empty_clusters:
            return adjusted

        ranked_indices = torch.argsort(min_distances, descending=True).tolist()
        used_indices: set[int] = set()

        for empty_cluster in empty_clusters:
            sample_index = self._select_reseed_sample(
                labels=adjusted,
                counts=counts,
                ranked_indices=ranked_indices,
                used_indices=used_indices,
            )
            if sample_index is None:
                break

            donor_cluster = int(adjusted[sample_index].item())
            adjusted[sample_index] = empty_cluster
            counts[donor_cluster] -= 1
            counts[empty_cluster] += 1
            used_indices.add(sample_index)

        return adjusted

    def _select_reseed_sample(
        self,
        labels: torch.Tensor,
        counts: torch.Tensor,
        ranked_indices: Sequence[int],
        used_indices: set[int],
    ) -> int | None:
        for sample_index in ranked_indices:
            if sample_index in used_indices:
                continue
            donor_cluster = int(labels[sample_index].item())
            if counts[donor_cluster] <= 1:
                continue
            return sample_index
        return None

    def _select_representatives(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        centroids: torch.Tensor,
    ) -> list[int]:
        """为每个簇选择离质心最近的样本作为代表。"""
        representatives: list[int] = []
        for cluster_id in range(centroids.size(0)):
            member_indices = (labels == cluster_id).nonzero(as_tuple=False).flatten()
            if member_indices.numel() == 0:
                raise RuntimeError(f"cluster {cluster_id} has no members")

            member_features = features[member_indices]
            centroid = centroids[cluster_id]
            distances = torch.norm(member_features - centroid, dim=1)
            best_local_index = int(distances.argmin().item())
            representatives.append(int(member_indices[best_local_index].item()))
        return representatives

    def _feature_method_name(self) -> str:
        """返回当前聚类使用的特征方法名。"""
        configured_method = getattr(self.config, "method", None)
        if isinstance(configured_method, str) and configured_method:
            return configured_method
        clusterer_name = self.__class__.__name__
        suffix = "Clusterer"
        if clusterer_name.endswith(suffix):
            clusterer_name = clusterer_name[: -len(suffix)]
        return clusterer_name.lower()


def resolve_model_output_field(model_outputs: Mapping[str, Any], key: str) -> Any:
    """统一从 model_outputs / metadata / 属性中读取字段。"""
    if key in model_outputs:
        return model_outputs[key]
    metadata = model_outputs.get("metadata")
    if isinstance(metadata, Mapping) and key in metadata:
        return metadata[key]
    return getattr(model_outputs, key, None)
