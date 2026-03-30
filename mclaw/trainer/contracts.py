from __future__ import annotations

"""训练阶段与外部后端解耦的本地协议。"""

from typing import Any, Mapping, Protocol, runtime_checkable

from mclaw.core.contracts import ActorBatch, AuxiliaryBatch


@runtime_checkable
class ActorBackendProtocol(Protocol):
    """兼容 PPO actor worker 的最小接口。"""

    def compute_log_prob(self, batch: ActorBatch) -> Mapping[str, Any]:
        """重算 actor log-prob，并返回可写回 `batch.metadata["old_log_probs"]` 的结果。"""

    def update_policy(self, batch: ActorBatch) -> Mapping[str, float]:
        """执行 PPO actor 更新。

        默认约定:
        - `batch.metadata["old_log_probs"]`：actor 重算的旧策略 log-prob
        - `batch.metadata["ref_log_probs"]`：reference policy log-prob
        - `batch.metadata["auxiliary_batch"]` / `["auxiliary_loss"]`：叠加到主 loss 的辅助监督
        - `old_log_probs` / `ref_log_probs` 是 rollout-time 固定信号，`update_policy()`
          不应在 PPO 多 epoch 过程中覆盖或就地修改它们
        """

    def update_aux_loss(self, batch: AuxiliaryBatch) -> Mapping[str, float]:
        """执行独立 auxiliary loss 更新的兼容接口（默认 MClaw train_step 不再调用）。"""


@runtime_checkable
class ReferencePolicyProtocol(Protocol):
    """兼容 reference policy 的最小接口。"""

    def compute_ref_log_prob(self, batch: ActorBatch) -> Mapping[str, Any]:
        """重算 reference log-prob，并返回可写回 `batch.metadata["ref_log_probs"]` 的结果。"""


@runtime_checkable
class LoggerProtocol(Protocol):
    """兼容日志后端的最小接口。"""

    def log(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        """记录训练指标。"""
