from __future__ import annotations

"""MClaw 训练循环接口。"""

from copy import deepcopy
from collections.abc import Sequence as SequenceABC
from contextlib import nullcontext
from collections.abc import Mapping as MappingABC, MutableMapping
from numbers import Real
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
        rollout_sharding_manager: Any | None = None,
        training_sharding_manager: Any | None = None,
    ) -> None:
        self.config = config
        self.tree_rollout = tree_rollout
        self.actor = actor
        self.ref_policy = ref_policy
        self.q_critic = q_critic
        self.logger = logger
        self.rollout_sharding_manager = rollout_sharding_manager
        self.training_sharding_manager = training_sharding_manager
        self.global_step = 0

    def build_rollout_engine(self) -> TreeRollout:
        """组装 TreeRollout 及其依赖模块。"""
        raise NotImplementedError("TODO: 实现 rollout 引擎组装逻辑。")

    def fit(self) -> None:
        """执行完整训练循环。"""
        # 计划中的主循环应当与 AgentGym-RL / Ray worker 语义对齐：
        # 1. 构建或恢复 dataloader / checkpoint / global_step。
        # 2. 对每个 prompt batch:
        #    - 进入 rollout sharding context，同步 rollout 权重并执行 tree rollout。
        #    - 切到 training sharding context（若未单独配置，则至少保证 FSDP actor 仍可前向）。
        #    - 先重算 actor old_log_probs 和 ref_log_probs，再执行多 epoch PPO update。
        #    - Q-head update 也必须在可用的 FSDP context 下运行，因为 q_critic.update()
        #      内部会重新做 backbone forward。
        #    - 记录 metrics，按 save_freq 处理 checkpoint。
        # 3. 支持 checkpoint 恢复后的 dataloader / step continuation。
        raise NotImplementedError(
            "TODO: implement fit() with dataloader iteration, checkpointing, and "
            "explicit rollout/training sharding-manager context switches."
        )

    def train_step(self, prompt_batch: Any) -> dict[str, float]:
        """执行单个 training step。"""
        if self.tree_rollout is None:
            raise ValueError("tree_rollout must be configured before calling train_step")

        if self._should_share_sharding_context():
            with self._rollout_context():
                rollout_output = self.tree_rollout.generate_tree_rollout(prompt_batch)
                metrics = self._build_rollout_metrics(rollout_output)
                metrics.update(
                    self.update_actor(
                        rollout_output.actor_data,
                        aux_actor_data=rollout_output.aux_actor_data,
                    )
                )
                metrics.update(self.update_q_head(rollout_output.critic_data))
        else:
            with self._rollout_context():
                rollout_output = self.tree_rollout.generate_tree_rollout(prompt_batch)

            metrics = self._build_rollout_metrics(rollout_output)
            with self._training_context():
                metrics.update(
                    self.update_actor(
                        rollout_output.actor_data,
                        aux_actor_data=rollout_output.aux_actor_data,
                    )
                )
                metrics.update(self.update_q_head(rollout_output.critic_data))

        metrics["trainer/step"] = float(self.global_step)

        if self.logger is not None:
            self.logger.log(metrics, step=self.global_step)
        self.global_step += 1
        return metrics

    def adapt_actor_batch(self, actor_data: ActorBatch) -> Any:
        """将本地 ActorBatch 转成外部后端可接受的格式。

        如果适配层返回的是一个新对象而不是原始 `ActorBatch`，它必须保留
        `metadata["old_log_probs"]`、`metadata["ref_log_probs"]` 以及
        `metadata["auxiliary_*"]` 这些训练期必需字段。
        """
        return actor_data

    def adapt_auxiliary_batch(self, aux_actor_data: AuxiliaryBatch) -> Any:
        """将本地 AuxiliaryBatch 转成外部后端可接受的格式。"""
        return aux_actor_data

    def adapt_critic_batch(self, critic_data: CriticBatch) -> Any:
        """将本地 CriticBatch 转成 Q-head 更新所需格式。"""
        return critic_data

    def update_actor(
        self,
        actor_data: ActorBatch,
        aux_actor_data: AuxiliaryBatch | None = None,
    ) -> Mapping[str, float]:
        """执行 PPO actor 更新。"""
        if self.actor is None or not actor_data.trajectories:
            return {}

        self.merge_auxiliary_into_actor_batch(actor_data, aux_actor_data)

        metrics: dict[str, float] = {}
        metrics.update(self.compute_old_log_probs(actor_data))
        metrics.update(self.compute_ref_log_probs(actor_data))
        frozen_training_signals = self._snapshot_actor_training_signals(actor_data)

        ppo_epochs = self._resolve_ppo_epochs()
        epoch_outputs: list[Mapping[str, float]] = []
        for _ in range(ppo_epochs):
            self._restore_actor_training_signals(actor_data, frozen_training_signals)
            adapted_batch = self.adapt_actor_batch(actor_data)
            self._apply_training_signals_to_adapted_batch(adapted_batch, frozen_training_signals)
            epoch_outputs.append(dict(self.actor.update_policy(adapted_batch)))

        self._restore_actor_training_signals(actor_data, frozen_training_signals)
        metrics.update(self._aggregate_float_metrics(epoch_outputs))
        metrics["actor/ppo_epochs"] = float(ppo_epochs)
        return metrics

    def merge_auxiliary_into_actor_batch(
        self,
        actor_data: ActorBatch,
        aux_actor_data: AuxiliaryBatch | None,
    ) -> None:
        """将 auxiliary 样本作为主 actor loss 的附加监督信息挂到 batch 上。"""
        aux_batch = aux_actor_data or AuxiliaryBatch()
        actor_data.metadata["auxiliary_batch"] = aux_batch
        actor_data.metadata["auxiliary_samples"] = list(aux_batch.samples)
        actor_data.metadata["auxiliary_loss"] = {
            "coef": float(self.config.aux_loss.coef),
            "use_same_advantage": bool(self.config.aux_loss.use_same_advantage),
            "weighting": str(self.config.aux_loss.weighting),
        }

    def compute_old_log_probs(self, actor_data: ActorBatch) -> Mapping[str, float]:
        """重算 PPO old log-prob，并把结果挂回 ActorBatch.metadata。"""
        if self.actor is None or not actor_data.trajectories:
            return {}

        adapted_batch = self.adapt_actor_batch(actor_data)
        output = dict(self.actor.compute_log_prob(adapted_batch))
        self._record_batch_signal(
            actor_data,
            signal_name="old_log_probs",
            output=output,
            preferred_keys=("old_log_probs", "log_probs"),
        )
        return self._extract_scalar_metrics(output, prefix="actor_log_prob/")

    def compute_ref_log_probs(self, actor_data: ActorBatch) -> Mapping[str, float]:
        """重算 reference policy log-prob，并把结果挂回 ActorBatch.metadata。"""
        if self.ref_policy is None or not actor_data.trajectories:
            return {}

        adapted_batch = self.adapt_actor_batch(actor_data)
        output = dict(self.ref_policy.compute_ref_log_prob(adapted_batch))
        self._record_batch_signal(
            actor_data,
            signal_name="ref_log_probs",
            output=output,
            preferred_keys=("ref_log_probs", "log_probs"),
        )
        return self._extract_scalar_metrics(output, prefix="ref_log_prob/")

    def update_auxiliary_loss(self, aux_actor_data: AuxiliaryBatch) -> Mapping[str, float]:
        """兼容旧后端的独立 auxiliary update 路径；默认 train_step 不再调用。"""
        if self.actor is None or not aux_actor_data.samples:
            return {}
        adapted_batch = self.adapt_auxiliary_batch(aux_actor_data)
        output = self.actor.update_aux_loss(adapted_batch)
        return dict(output)

    def update_q_head(self, critic_data: CriticBatch) -> Mapping[str, float]:
        """执行 Q-head TD 更新。"""
        if self.q_critic is None or not critic_data.samples:
            return {}
        update_steps = max(int(getattr(self.q_critic.config, "update_freq", 1)), 1)
        outputs: list[Mapping[str, float]] = []
        for _ in range(update_steps):
            adapted_batch = self.adapt_critic_batch(critic_data)
            outputs.append(dict(self.q_critic.update(adapted_batch)))

        metrics = self._aggregate_float_metrics(outputs)
        metrics["critic/update_steps"] = float(update_steps)
        return metrics

    def save_checkpoint(self, step: int) -> None:
        """保存训练状态。"""
        raise NotImplementedError("TODO: 实现 checkpoint 保存逻辑。")

    def _record_batch_signal(
        self,
        actor_data: ActorBatch,
        signal_name: str,
        output: Mapping[str, Any],
        preferred_keys: tuple[str, ...],
    ) -> None:
        actor_data.metadata[f"{signal_name}_output"] = dict(output)
        payload = self._extract_primary_payload(output, preferred_keys)
        actor_data.metadata[signal_name] = payload
        self._assign_trajectory_signal(actor_data, signal_name, payload)

    def _extract_primary_payload(
        self,
        output: Mapping[str, Any],
        preferred_keys: tuple[str, ...],
    ) -> Any:
        for key in preferred_keys:
            if key in output:
                return output[key]
        return dict(output)

    def _extract_scalar_metrics(
        self,
        output: Mapping[str, Any],
        prefix: str,
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for key, value in output.items():
            metric_value = self._to_float_metric(value)
            if metric_value is None:
                continue
            metrics[f"{prefix}{key}"] = metric_value
        return metrics

    def _aggregate_float_metrics(
        self,
        outputs: list[Mapping[str, float]],
    ) -> dict[str, float]:
        if not outputs:
            return {}

        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for output in outputs:
            for key, value in output.items():
                metric_value = self._to_float_metric(value)
                if metric_value is None:
                    continue
                sums[key] = sums.get(key, 0.0) + metric_value
                counts[key] = counts.get(key, 0) + 1
        return {
            key: sums[key] / counts[key]
            for key in sums
            if counts.get(key, 0) > 0
        }

    def _to_float_metric(self, value: Any) -> float | None:
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, Real):
            return float(value)
        item = getattr(value, "item", None)
        if callable(item):
            try:
                scalar = item()
            except (TypeError, ValueError):
                return None
            if isinstance(scalar, bool):
                return float(scalar)
            if isinstance(scalar, Real):
                return float(scalar)
        return None

    def _resolve_ppo_epochs(self) -> int:
        raw_value = self._resolve_nested_config_value(
            self.config.actor_rollout_ref,
            ("actor", "ppo_epochs"),
            default=self._resolve_nested_config_value(
                self.config.actor_rollout_ref,
                ("ppo_epochs",),
                default=1,
            ),
        )
        try:
            return max(int(raw_value), 1)
        except (TypeError, ValueError):
            return 1

    def _rollout_context(self) -> Any:
        return self._optional_context(self.rollout_sharding_manager)

    def _training_context(self) -> Any:
        return self._optional_context(self.training_sharding_manager)

    def _optional_context(self, manager: Any) -> Any:
        if manager is None:
            return nullcontext()
        if hasattr(manager, "__enter__") and hasattr(manager, "__exit__"):
            return manager
        raise TypeError("sharding manager must be a context manager when provided")

    def _build_rollout_metrics(self, rollout_output: Any) -> dict[str, float]:
        return {
            "rollout/n_roots": float(len(rollout_output.roots)),
            "rollout/n_aux_samples": float(len(rollout_output.aux_actor_data.samples)),
            "rollout/n_critic_samples": float(len(rollout_output.critic_data.samples)),
            "rollout/n_actor_trajectories": float(len(rollout_output.actor_data.trajectories)),
        }

    def _snapshot_actor_training_signals(self, actor_data: ActorBatch) -> dict[str, Any]:
        return {
            "old_log_probs": deepcopy(actor_data.metadata.get("old_log_probs")),
            "ref_log_probs": deepcopy(actor_data.metadata.get("ref_log_probs")),
            "auxiliary_batch": deepcopy(actor_data.metadata.get("auxiliary_batch")),
            "auxiliary_samples": deepcopy(actor_data.metadata.get("auxiliary_samples")),
            "auxiliary_loss": deepcopy(actor_data.metadata.get("auxiliary_loss")),
        }

    def _restore_actor_training_signals(
        self,
        actor_data: ActorBatch,
        frozen_training_signals: Mapping[str, Any],
    ) -> None:
        for key, value in frozen_training_signals.items():
            actor_data.metadata[key] = deepcopy(value)

    def _apply_training_signals_to_adapted_batch(
        self,
        adapted_batch: Any,
        frozen_training_signals: Mapping[str, Any],
    ) -> None:
        if adapted_batch is None:
            raise TypeError("adapt_actor_batch must not return None")

        if hasattr(adapted_batch, "metadata"):
            metadata = getattr(adapted_batch, "metadata")
            if isinstance(metadata, MutableMapping):
                for key, value in frozen_training_signals.items():
                    metadata[key] = deepcopy(value)
                return

        if isinstance(adapted_batch, MutableMapping):
            metadata = adapted_batch.get("metadata")
            if metadata is None:
                metadata = {}
                adapted_batch["metadata"] = metadata
            if isinstance(metadata, MutableMapping):
                for key, value in frozen_training_signals.items():
                    metadata[key] = deepcopy(value)
                return

        if isinstance(adapted_batch, ActorBatch):
            for key, value in frozen_training_signals.items():
                adapted_batch.metadata[key] = deepcopy(value)
            return

        raise TypeError(
            "adapt_actor_batch must return an ActorBatch or an object exposing a mutable "
            "metadata mapping so old/ref log probs and auxiliary signals survive PPO epochs"
        )

    def _resolve_nested_config_value(
        self,
        container: Any,
        path: tuple[str, ...],
        default: Any,
    ) -> Any:
        current = container
        for key in path:
            if current is None:
                return default
            if isinstance(current, MappingABC):
                if key not in current:
                    return default
                current = current[key]
                continue
            if hasattr(current, key):
                current = getattr(current, key)
                continue
            getter = getattr(current, "get", None)
            if callable(getter):
                sentinel = object()
                try:
                    candidate = getter(key, sentinel)
                except TypeError:
                    try:
                        candidate = getter(key)
                    except Exception:
                        return default
                except Exception:
                    return default
                if candidate is sentinel:
                    return default
                current = candidate
                continue
            return default
        return current

    def _should_share_sharding_context(self) -> bool:
        if self.rollout_sharding_manager is None:
            return False
        if self.training_sharding_manager is None:
            return True
        return self.training_sharding_manager is self.rollout_sharding_manager

    def _assign_trajectory_signal(
        self,
        actor_data: ActorBatch,
        signal_name: str,
        payload: Any,
    ) -> None:
        if signal_name not in {"old_log_probs", "ref_log_probs"}:
            return

        per_trajectory_values = self._coerce_per_trajectory_signal(
            actor_data=actor_data,
            payload=payload,
        )
        if per_trajectory_values is None:
            return

        attribute_name = signal_name
        for trajectory, values in zip(actor_data.trajectories, per_trajectory_values):
            setattr(trajectory, attribute_name, values)

    def _coerce_per_trajectory_signal(
        self,
        actor_data: ActorBatch,
        payload: Any,
    ) -> list[list[float]] | None:
        sequence = self._to_sequence(payload)
        if sequence is None or len(sequence) != len(actor_data.trajectories):
            return None

        normalized: list[list[float]] = []
        for trajectory, item in zip(actor_data.trajectories, sequence):
            values = self._coerce_float_sequence(item)
            if values is None:
                return None
            if len(values) == len(trajectory.input_ids):
                normalized.append(values)
                continue
            response_token_count = sum(int(flag) for flag in trajectory.response_mask)
            if len(values) == response_token_count:
                scattered = [0.0] * len(trajectory.input_ids)
                value_index = 0
                for token_index, is_response in enumerate(trajectory.response_mask):
                    if not is_response:
                        continue
                    scattered[token_index] = values[value_index]
                    value_index += 1
                normalized.append(scattered)
                continue
            return None
        return normalized

    def _coerce_float_sequence(self, value: Any) -> list[float] | None:
        if isinstance(value, bool):
            return [float(value)]
        if isinstance(value, Real):
            return [float(value)]

        sequence = self._to_sequence(value)
        if sequence is None:
            return None

        normalized: list[float] = []
        for item in sequence:
            if isinstance(item, bool):
                normalized.append(float(item))
                continue
            if isinstance(item, Real):
                normalized.append(float(item))
                continue
            item_fn = getattr(item, "item", None)
            if callable(item_fn):
                try:
                    scalar = item_fn()
                except (TypeError, ValueError):
                    return None
                if isinstance(scalar, bool):
                    normalized.append(float(scalar))
                    continue
                if isinstance(scalar, Real):
                    normalized.append(float(scalar))
                    continue
            return None
        return normalized

    def _to_sequence(self, value: Any) -> list[Any] | None:
        if isinstance(value, SequenceABC) and not isinstance(value, (str, bytes, bytearray)):
            return list(value)
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            try:
                converted = tolist()
            except (TypeError, ValueError):
                return None
            if isinstance(converted, SequenceABC) and not isinstance(converted, (str, bytes, bytearray)):
                return list(converted)
        return None
