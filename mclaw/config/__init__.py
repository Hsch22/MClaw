from __future__ import annotations

"""MClaw 配置类型定义。"""

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

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
    grad_clip_norm: float | None = 1.0
    micro_batch_size: int = 32


@dataclass(slots=True)
class AuxLossConfig:
    """Auxiliary loss 超参。"""

    coef: float = 0.2
    use_same_advantage: bool = True
    weighting: str = "equal_per_selected_cluster"


@dataclass(slots=True)
class AdapterConfig:
    """外部系统适配层配置。"""

    actor_backend: str = "verl"
    ref_policy: str = "verl"
    inference_engine: str = "verl_vllm"
    env_client: str = "agentenv"
    rollout_handler: str = "verl"
    logger: str = "standard"
    task_name: str = ""
    env_addr: str = ""
    max_retries: int = 3


@dataclass(slots=True)
class ModelConfig:
    """基础模型与 tokenizer 配置。"""

    family: str = ""
    model_path: str = ""
    tokenizer_path: str = ""
    dtype: str = "bfloat16"
    trust_remote_code: bool = True
    revision: str | None = None
    use_remove_padding: bool = False
    pad_token_as_eos: bool = True


@dataclass(slots=True)
class DistributedConfig:
    """分布式/FSDP/vLLM 相关配置。"""

    enable_fsdp: bool = False
    tensor_parallel_size: int = 1
    full_params: bool = False
    device: str = "cuda"
    fsdp_config: dict[str, Any] = field(default_factory=dict)
    use_rollout_sharding_manager: bool = True
    use_training_sharding_manager: bool = False


@dataclass(slots=True)
class DataConfig:
    """训练数据加载配置。"""

    train_file: str = ""
    train_batch_size: int = 1
    shuffle: bool = True
    num_workers: int = 0
    drop_last: bool = False
    prompt_key: str = "prompt"
    max_prompt_length: int = 1024
    max_response_length: int = 512
    use_verl_dataset: bool = False
    cache_dir: str = ""


@dataclass(slots=True)
class TrainerRuntimeConfig:
    """训练循环与 checkpoint 配置。"""

    total_epochs: int = 1
    max_steps: int = 0
    save_freq: int = 0
    default_local_dir: str = "MClaw/checkpoints"
    checkpoint_dir: str = ""
    resume_from: str = ""
    seed: int = 0


@dataclass(slots=True)
class EnvironmentConfig:
    """环境交互配置。"""

    adapter: str = "agentgym-rl-qwen3"
    instance_pool: int = 16
    reset_kwargs: dict[str, Any] = field(default_factory=dict)
    step_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LoggingConfig:
    """日志与追踪配置。"""

    path_pattern: str = "MClaw/log/{timestamp}.log"
    level: str = "INFO"
    tracker: str = "none"
    project_name: str = ""
    experiment_name: str = ""
    tracker_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MClawTrainerConfig:
    """训练器级别的配置聚合。"""

    tree_rollout: TreeRolloutConfig = field(default_factory=TreeRolloutConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    q_critic: QCriticConfig = field(default_factory=QCriticConfig)
    aux_loss: AuxLossConfig = field(default_factory=AuxLossConfig)
    algorithm: dict[str, Any] = field(default_factory=dict)
    actor_rollout_ref: dict[str, Any] = field(default_factory=dict)
    data: DataConfig = field(default_factory=DataConfig)
    trainer: TrainerRuntimeConfig = field(default_factory=TrainerRuntimeConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def trainer_config_from_mapping(value: Mapping[str, Any]) -> MClawTrainerConfig:
    """从普通 mapping 构造强类型 `MClawTrainerConfig`。"""
    config = dict(value)
    return MClawTrainerConfig(
        tree_rollout=_build_dataclass(TreeRolloutConfig, config.get("tree_rollout")),
        clustering=_build_clustering_config(config.get("clustering")),
        q_critic=_build_dataclass(QCriticConfig, config.get("q_critic")),
        aux_loss=_build_dataclass(AuxLossConfig, config.get("aux_loss")),
        algorithm=_as_plain_dict(config.get("algorithm")),
        actor_rollout_ref=_as_plain_dict(config.get("actor_rollout_ref")),
        data=_build_dataclass(DataConfig, config.get("data")),
        trainer=_build_dataclass(TrainerRuntimeConfig, config.get("trainer")),
        environment=_build_dataclass(EnvironmentConfig, config.get("environment")),
        adapter=_build_dataclass(AdapterConfig, config.get("adapter")),
        model=_build_dataclass(ModelConfig, config.get("model")),
        distributed=_build_dataclass(DistributedConfig, config.get("distributed")),
        logging=_build_dataclass(LoggingConfig, config.get("logging")),
    )


def _build_clustering_config(value: Mapping[str, Any] | None) -> ClusteringConfig:
    source = _as_plain_dict(value)
    defaults = ClusteringConfig()
    return ClusteringConfig(
        method=str(source.get("method", defaults.method)),
        pca_dim=int(source.get("pca_dim", defaults.pca_dim)),
        hidden_state=_build_dataclass(HiddenStateClusterConfig, source.get("hidden_state")),
        output_grad=_build_dataclass(OutputGradClusterConfig, source.get("output_grad")),
        logit_distribution=_build_dataclass(
            LogitDistributionConfig,
            source.get("logit_distribution"),
        ),
    )


def _build_dataclass(cls: type[Any], value: Mapping[str, Any] | None) -> Any:
    source = _as_plain_dict(value)
    allowed_keys = {item.name for item in fields(cls)}
    filtered = {
        key: item
        for key, item in source.items()
        if key in allowed_keys
    }
    return cls(**filtered)


def _as_plain_dict(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    return dict(value)


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "AdapterConfig",
    "AuxLossConfig",
    "ClusteringConfig",
    "DataConfig",
    "DistributedConfig",
    "EnvironmentConfig",
    "HiddenStateClusterConfig",
    "LoggingConfig",
    "LogitDistributionConfig",
    "MClawTrainerConfig",
    "ModelConfig",
    "OutputGradClusterConfig",
    "QCriticConfig",
    "TrainerRuntimeConfig",
    "TreeRolloutConfig",
    "trainer_config_from_mapping",
]
