from __future__ import annotations

"""MClaw 训练循环接口。"""

from typing import Any, Mapping

from mclaw.config import MClawTrainerConfig
from mclaw.core import ActorBatch, AuxiliaryBatch, CriticBatch, TreeRollout
from mclaw.critic import QCritic

from .contracts import ActorBackendProtocol, LoggerProtocol, ReferencePolicyProtocol


class MClawTrainer:
    """串起 rollout、actor 更新、auxiliary loss 和 Q-head 更新。"""

    def __init__(
        self,
        config: MClawTrainerConfig,
        tree_rollout: TreeRollout | None = None,
        actor: ActorBackendProtocol | None = None,
        ref_policy: ReferencePolicyProtocol | None = None,
        q_critic: QCritic | None = None,
        logger: LoggerProtocol | None = None,
    ) -> None:
        self.config = config
        self.tree_rollout = tree_rollout
        self.actor = actor
        self.ref_policy = ref_policy
        self.q_critic = q_critic
        self.logger = logger

    def build_rollout_engine(self) -> TreeRollout:
        """组装 TreeRollout 及其依赖模块。"""
        raise NotImplementedError("TODO: 实现 rollout 引擎组装逻辑。")

    def fit(self) -> None:
        """执行完整训练循环。"""
        raise NotImplementedError("TODO: 实现训练主循环。")

    def train_step(self, prompt_batch: Any) -> dict[str, float]:
        """执行单个 training step。"""
        raise NotImplementedError("TODO: 实现单步训练逻辑。")

    def adapt_actor_batch(self, actor_data: ActorBatch) -> Any:
        """将本地 ActorBatch 转成外部后端可接受的格式。"""
        raise NotImplementedError("TODO: 实现 ActorBatch 适配逻辑。")

    def adapt_auxiliary_batch(self, aux_actor_data: AuxiliaryBatch) -> Any:
        """将本地 AuxiliaryBatch 转成外部后端可接受的格式。"""
        raise NotImplementedError("TODO: 实现 AuxiliaryBatch 适配逻辑。")

    def adapt_critic_batch(self, critic_data: CriticBatch) -> Any:
        """将本地 CriticBatch 转成 Q-head 更新所需格式。"""
        raise NotImplementedError("TODO: 实现 CriticBatch 适配逻辑。")

    def update_actor(self, actor_data: ActorBatch) -> Mapping[str, float]:
        """执行 PPO actor 更新。"""
        raise NotImplementedError("TODO: 实现 actor 更新逻辑。")

    def update_auxiliary_loss(self, aux_actor_data: AuxiliaryBatch) -> Mapping[str, float]:
        """执行同簇 siblings 的 auxiliary loss 更新。"""
        raise NotImplementedError("TODO: 实现 auxiliary loss 更新逻辑。")

    def update_q_head(self, critic_data: CriticBatch) -> Mapping[str, float]:
        """执行 Q-head TD 更新。"""
        raise NotImplementedError("TODO: 实现 Q-head 更新调度逻辑。")

    def save_checkpoint(self, step: int) -> None:
        """保存训练状态。"""
        raise NotImplementedError("TODO: 实现 checkpoint 保存逻辑。")
