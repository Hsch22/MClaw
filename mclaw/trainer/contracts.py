from __future__ import annotations

"""训练阶段与外部后端解耦的本地协议。"""

from typing import Any, Mapping, Protocol, runtime_checkable

from mclaw.core.contracts import ActorBatch, AuxiliaryBatch


@runtime_checkable
class ActorBackendProtocol(Protocol):
    """兼容 PPO actor worker 的最小接口。"""

    def compute_log_prob(self, batch: ActorBatch) -> Mapping[str, Any]:
        """重算 actor log-prob。"""

    def update_policy(self, batch: ActorBatch) -> Mapping[str, float]:
        """执行 PPO actor 更新。"""

    def update_aux_loss(self, batch: AuxiliaryBatch) -> Mapping[str, float]:
        """执行 auxiliary loss 更新。"""


@runtime_checkable
class ReferencePolicyProtocol(Protocol):
    """兼容 reference policy 的最小接口。"""

    def compute_ref_log_prob(self, batch: ActorBatch) -> Mapping[str, Any]:
        """重算 reference log-prob。"""


@runtime_checkable
class LoggerProtocol(Protocol):
    """兼容日志后端的最小接口。"""

    def log(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        """记录训练指标。"""
