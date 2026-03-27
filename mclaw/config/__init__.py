from __future__ import annotations

"""MClaw 配置类型定义。"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path(__file__).with_name("mclaw_trainer.yaml")


@dataclass(slots=True)
class HiddenStateClusterConfig:
    """Hidden-state 聚类配置。"""

    layer: int = -1


@dataclass(slots=True)
class OutputGradClusterConfig:
    """Output-gradient 聚类配置。"""

    use_mean_pooling: bool = True
    top_k_logprobs: int = 200


@dataclass(slots=True)
class LogitDistributionConfig:
    """Logit 分布聚类配置。"""

    top_k: int = 100


@dataclass(slots=True)
class TreeRolloutConfig:
    """树状 rollout 超参。"""

    root_budget: int = 256
    n_envs: int = 16
    root_clusters: int = 16
    branch_budget: int = 16
    intra_branch_clusters: int = 4
    max_rounds: int = 30


@dataclass(slots=True)
class ClusteringConfig:
    """聚类模块超参。"""

    method: str = "hidden_state"
    pca_dim: int = 128
    hidden_state: HiddenStateClusterConfig = field(default_factory=HiddenStateClusterConfig)
    output_grad: OutputGradClusterConfig = field(default_factory=OutputGradClusterConfig)
    logit_distribution: LogitDistributionConfig = field(default_factory=LogitDistributionConfig)


@dataclass(slots=True)
class QCriticConfig:
    """Q-critic 训练超参。"""

    hidden_dim: int = 3584
    intermediate_dim: int = 1024
    lr: float = 1e-4
    gamma: float = 0.99
    update_freq: int = 1
    td_lambda: float = 0.95


@dataclass(slots=True)
class AuxLossConfig:
    """Auxiliary loss 超参。"""

    coef: float = 0.2
    use_same_advantage: bool = True
    weighting: str = "equal_per_selected_cluster"


@dataclass(slots=True)
class MClawTrainerConfig:
    """训练器级别的配置聚合。"""

    tree_rollout: TreeRolloutConfig = field(default_factory=TreeRolloutConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    q_critic: QCriticConfig = field(default_factory=QCriticConfig)
    aux_loss: AuxLossConfig = field(default_factory=AuxLossConfig)
    algorithm: dict[str, Any] = field(default_factory=dict)
    actor_rollout_ref: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    trainer: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    logging: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "AuxLossConfig",
    "ClusteringConfig",
    "HiddenStateClusterConfig",
    "LogitDistributionConfig",
    "MClawTrainerConfig",
    "OutputGradClusterConfig",
    "QCriticConfig",
    "TreeRolloutConfig",
]
