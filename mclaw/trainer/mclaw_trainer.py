from __future__ import annotations

"""MClaw 训练循环接口。"""

from collections.abc import Mapping as MappingABC, MutableMapping
from collections.abc import Sequence as SequenceABC
from copy import deepcopy
from dataclasses import asdict
import json
from contextlib import nullcontext
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

from mclaw.adapters import AdaptedActorBatch, DataProtoAdapter
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
        *,
        tokenizer: Any | None = None,
        inference_engine: Any | None = None,
        clusterer: Any | None = None,
        env_client_factory: Any | None = None,
        branch_selector: Any | None = None,
        rollout_handler_factory: Any | None = None,
        dataproto_adapter: DataProtoAdapter | None = None,
    ) -> None:
        self.config = config
        self.tree_rollout = tree_rollout
        self.actor = actor
        self.ref_policy = ref_policy
        self.q_critic = q_critic
        self.logger = logger
        self.rollout_sharding_manager = rollout_sharding_manager
        self.training_sharding_manager = training_sharding_manager
        self.tokenizer = tokenizer
        self.inference_engine = inference_engine
        self.clusterer = clusterer
        self.env_client_factory = env_client_factory
        self.branch_selector = branch_selector
        self.rollout_handler_factory = rollout_handler_factory
        self.dataproto_adapter = dataproto_adapter
        self.global_step = 0
        self._current_epoch = 0
        self._current_batch_index = -1

    def build_rollout_engine(self) -> TreeRollout:
        """组装 TreeRollout 及其依赖模块。"""
        if self.tree_rollout is not None:
            return self.tree_rollout
        if self.inference_engine is None:
            raise ValueError("inference_engine must be configured before building TreeRollout")
        if self.q_critic is None:
            raise ValueError("q_critic must be configured before building TreeRollout")
        if self.clusterer is None:
            raise ValueError("clusterer must be configured before building TreeRollout")
        if self.tokenizer is None:
            raise ValueError("tokenizer must be configured before building TreeRollout")

        self.tree_rollout = TreeRollout(
            inference_engine=self.inference_engine,
            actor_module_fsdp=self.q_critic.actor_module_fsdp,
            q_critic=self.q_critic,
            clusterer=self.clusterer,
            tokenizer=self.tokenizer,
            config=self.config.tree_rollout,
            env_client_factory=self.env_client_factory,
            branch_selector=self.branch_selector,
            handler_factory=self.rollout_handler_factory,
        )
        return self.tree_rollout

    def fit(self) -> None:
        """执行完整训练循环。"""
        self.build_rollout_engine()
        dataloader = self._build_dataloader()
        resume_state = self._load_checkpoint_if_needed()

        total_epochs = max(int(self.config.trainer.total_epochs), 1)
        max_steps = max(int(self.config.trainer.max_steps), 0)
        start_epoch = int(resume_state.get("epoch", 0))
        start_batch_index = int(resume_state.get("batch_index", -1))

        if max_steps > 0 and self.global_step >= max_steps:
            return

        for epoch in range(start_epoch, total_epochs):
            skip_completed_batches = epoch == start_epoch
            for batch_index, prompt_batch in enumerate(dataloader):
                if skip_completed_batches and batch_index <= start_batch_index:
                    continue

                normalized_prompt_batch = self._normalize_prompt_batch(prompt_batch)
                self.train_step(normalized_prompt_batch)

                self._current_epoch = epoch
                self._current_batch_index = batch_index

                save_freq = max(int(self.config.trainer.save_freq), 0)
                if save_freq > 0 and self.global_step > 0 and self.global_step % save_freq == 0:
                    self.save_checkpoint(self.global_step)

                if max_steps > 0 and self.global_step >= max_steps:
                    return

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
        """将本地 ActorBatch 转成外部后端可接受的格式。"""
        if self.dataproto_adapter is None:
            return actor_data
        return self.dataproto_adapter.adapt_actor_batch(
            actor_data,
            include_ref_log_prob=self._should_include_ref_log_prob(),
            meta_info_overrides=self._build_dataproto_meta_info(),
        )

    def adapt_auxiliary_batch(self, aux_actor_data: AuxiliaryBatch) -> Any:
        """将本地 AuxiliaryBatch 转成外部后端可接受的格式。"""
        return aux_actor_data

    def adapt_critic_batch(self, critic_data: CriticBatch) -> Any:
        """将本地 CriticBatch 转成 Q-head 更新所需格式。"""
        return list(critic_data.samples)

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
        self._restore_actor_training_signals(actor_data, frozen_training_signals)
        adapted_batch = self.adapt_actor_batch(actor_data)
        self._apply_training_signals_to_adapted_batch(adapted_batch, frozen_training_signals)
        update_output = dict(self.actor.update_policy(adapted_batch))

        self._restore_actor_training_signals(actor_data, frozen_training_signals)
        metrics.update(update_output)
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
        """执行同簇 auxiliary policy gradient 更新。"""
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
        torch = _import_torch()

        checkpoint_dir = self._resolve_checkpoint_dir()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"global_step_{step}.pt"

        state = {
            "global_step": int(step),
            "epoch": int(self._current_epoch),
            "batch_index": int(self._current_batch_index),
            "config": asdict(self.config),
            "actor_module_state_dict": self._maybe_state_dict(self._resolve_actor_module()),
            "q_head_state_dict": self._maybe_state_dict(self._resolve_q_head()),
            "q_critic_optimizer_state_dict": self._maybe_optimizer_state_dict(
                getattr(self.q_critic, "optimizer", None)
            ),
            "actor_optimizer_state_dict": self._maybe_optimizer_state_dict(
                self._resolve_actor_optimizer(),
                module=self._resolve_actor_module(),
            ),
        }
        if self._is_primary_process():
            torch.save(state, checkpoint_path)
        self._maybe_distributed_barrier()

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

        if isinstance(adapted_batch, AdaptedActorBatch):
            return

        if hasattr(adapted_batch, "batch") and hasattr(adapted_batch, "meta_info"):
            return

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
            "adapt_actor_batch must return an ActorBatch, an AdaptedActorBatch, a "
            "DataProto-like object, or an object exposing mutable metadata"
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

    def _build_dataproto_meta_info(self) -> dict[str, Any]:
        actor_cfg = self._as_mapping(
            self._resolve_nested_config_value(
                self.config.actor_rollout_ref,
                ("actor",),
                default={},
            )
        )
        rollout_cfg = self._as_mapping(
            self._resolve_nested_config_value(
                self.config.actor_rollout_ref,
                ("rollout",),
                default={},
            )
        )
        return {
            "micro_batch_size": int(actor_cfg.get("ppo_micro_batch_size_per_gpu", 1)),
            "temperature": float(rollout_cfg.get("temperature", 1.0)),
            "use_dynamic_bsz": bool(actor_cfg.get("use_dynamic_bsz", False)),
        }

    def _should_include_ref_log_prob(self) -> bool:
        return bool(
            self._resolve_nested_config_value(
                self.config.actor_rollout_ref,
                ("actor", "use_kl_loss"),
                default=False,
            )
        )

    def _build_dataloader(self) -> Any:
        if self.tokenizer is None:
            raise ValueError("tokenizer must be configured before calling fit()")

        DataLoader = _import_dataloader()
        dataset = self._build_dataset()
        return DataLoader(
            dataset,
            batch_size=max(int(self.config.data.train_batch_size), 1),
            shuffle=bool(self.config.data.shuffle),
            num_workers=max(int(self.config.data.num_workers), 0),
            drop_last=bool(self.config.data.drop_last),
            collate_fn=_prompt_list_collate,
        )

    def _build_dataset(self) -> Any:
        if self.config.data.use_verl_dataset:
            return self._build_verl_dataset()

        train_file = self.config.data.train_file.strip()
        if not train_file:
            raise ValueError("data.train_file must be set before calling fit()")
        return _ListPromptDataset(_load_prompt_items_from_file(Path(train_file)))

    def _build_verl_dataset(self) -> Any:
        OmegaConf, RLHFDataset = _import_verl_dataset_components()
        data_config = OmegaConf.create(
            {
                "cache_dir": self.config.data.cache_dir,
                "prompt_key": self.config.data.prompt_key,
                "max_prompt_length": self.config.data.max_prompt_length,
                "max_response_length": self.config.data.max_response_length,
            }
        )
        agentgym_config = OmegaConf.create(
            {
                "task_name": self.config.adapter.task_name,
                "env_addr": self.config.adapter.env_addr,
                "max_retries": self.config.adapter.max_retries,
            }
        )
        return RLHFDataset(
            data_file=self.config.data.train_file,
            tokenizer=self.tokenizer,
            data_config=data_config,
            agentgym_config=agentgym_config,
        )

    def _normalize_prompt_batch(self, prompt_batch: Any) -> list[Any]:
        if isinstance(prompt_batch, list):
            items = prompt_batch
        elif isinstance(prompt_batch, tuple):
            items = list(prompt_batch)
        else:
            items = [prompt_batch]
        return [self._normalize_prompt_item(item) for item in items]

    def _normalize_prompt_item(self, item: Any) -> Any:
        if isinstance(item, MappingABC):
            normalized = {
                str(key): _to_python_value(value)
                for key, value in dict(item).items()
            }
            prompt_token_ids = normalized.get("prompt_token_ids")
            input_ids = normalized.get("input_ids")
            attention_mask = normalized.get("attention_mask")
            if prompt_token_ids is None and isinstance(input_ids, list):
                prompt_token_ids = [int(token_id) for token_id in input_ids]
                if isinstance(attention_mask, list) and len(attention_mask) == len(prompt_token_ids):
                    prompt_token_ids = [
                        token_id
                        for token_id, mask in zip(prompt_token_ids, attention_mask)
                        if int(mask) != 0
                    ]
                normalized["prompt_token_ids"] = prompt_token_ids
            if "item_id" not in normalized and "index" in normalized:
                normalized["item_id"] = normalized["index"]
            if "env_reset_kwargs" not in normalized and self.config.environment.reset_kwargs:
                normalized["env_reset_kwargs"] = dict(self.config.environment.reset_kwargs)
            return normalized
        return _to_python_value(item)

    def _load_checkpoint_if_needed(self) -> dict[str, Any]:
        checkpoint_path = self._resolve_resume_checkpoint()
        if checkpoint_path is None:
            return {}

        torch = _import_torch()
        try:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(checkpoint_path, map_location="cpu")
        self.global_step = int(state.get("global_step", 0))
        self._current_epoch = int(state.get("epoch", 0))
        self._current_batch_index = int(state.get("batch_index", -1))

        self._restore_state_dict(self._resolve_actor_module(), state.get("actor_module_state_dict"))
        self._restore_state_dict(self._resolve_q_head(), state.get("q_head_state_dict"))
        self._restore_optimizer_state(
            getattr(self.q_critic, "optimizer", None),
            state.get("q_critic_optimizer_state_dict"),
        )
        self._restore_optimizer_state(
            self._resolve_actor_optimizer(),
            state.get("actor_optimizer_state_dict"),
            module=self._resolve_actor_module(),
        )
        return state

    def _resolve_resume_checkpoint(self) -> Path | None:
        resume_from = self.config.trainer.resume_from.strip()
        if not resume_from:
            return None

        resume_path = Path(resume_from)
        if resume_path.is_file():
            return resume_path
        if resume_path.is_dir():
            candidates = sorted(
                resume_path.glob("global_step_*.pt"),
                key=_checkpoint_sort_key,
            )
            if not candidates:
                raise FileNotFoundError(f"no checkpoint files found under {resume_path}")
            return candidates[-1]
        raise FileNotFoundError(f"checkpoint path does not exist: {resume_path}")

    def _resolve_checkpoint_dir(self) -> Path:
        checkpoint_dir = self.config.trainer.checkpoint_dir.strip()
        if checkpoint_dir:
            return Path(checkpoint_dir)
        return Path(self.config.trainer.default_local_dir)

    def _resolve_actor_module(self) -> Any | None:
        if self.q_critic is not None and getattr(self.q_critic, "actor_module_fsdp", None) is not None:
            return self.q_critic.actor_module_fsdp
        actor_impl = getattr(self.actor, "actor", None)
        if actor_impl is not None:
            return getattr(actor_impl, "actor_module", None)
        return None

    def _resolve_q_head(self) -> Any | None:
        if self.q_critic is None:
            return None
        return getattr(self.q_critic, "q_head", None)

    def _resolve_actor_optimizer(self) -> Any | None:
        actor_impl = getattr(self.actor, "actor", None)
        if actor_impl is None:
            return None
        return getattr(actor_impl, "actor_optimizer", None)

    def _maybe_state_dict(self, module: Any) -> Any:
        if module is None or not hasattr(module, "state_dict"):
            return None
        fsdp_type = _try_import_fsdp()
        if fsdp_type is not None and isinstance(module, fsdp_type):
            return _get_fsdp_full_state_dict(module, rank0_only=True)
        return module.state_dict()

    def _maybe_optimizer_state_dict(self, optimizer: Any, module: Any | None = None) -> Any:
        if optimizer is None or not hasattr(optimizer, "state_dict"):
            return None
        fsdp_type = _try_import_fsdp()
        if (
            module is not None
            and fsdp_type is not None
            and isinstance(module, fsdp_type)
            and hasattr(fsdp_type, "optim_state_dict")
        ):
            return fsdp_type.optim_state_dict(module, optimizer)
        return optimizer.state_dict()

    def _restore_state_dict(self, module: Any, state_dict: Any) -> None:
        if module is None or state_dict is None:
            return
        if hasattr(module, "load_state_dict"):
            fsdp_type = _try_import_fsdp()
            if fsdp_type is not None and isinstance(module, fsdp_type):
                load_result = _load_fsdp_full_state_dict(module, state_dict)
            else:
                load_result = module.load_state_dict(state_dict)
            _validate_load_state_result(load_result, module=module)

    def _restore_optimizer_state(
        self,
        optimizer: Any,
        state_dict: Any,
        module: Any | None = None,
    ) -> None:
        if optimizer is None or state_dict is None:
            return
        if hasattr(optimizer, "load_state_dict"):
            fsdp_type = _try_import_fsdp()
            if (
                module is not None
                and fsdp_type is not None
                and isinstance(module, fsdp_type)
                and hasattr(fsdp_type, "optim_state_dict_to_load")
            ):
                state_dict = fsdp_type.optim_state_dict_to_load(
                    module,
                    optimizer,
                    state_dict,
                )
            optimizer.load_state_dict(state_dict)

    def _as_mapping(self, value: Any) -> Mapping[str, Any]:
        if isinstance(value, MappingABC):
            return value
        return {}

    def _is_primary_process(self) -> bool:
        torch = _import_torch()
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return True
        return int(torch.distributed.get_rank()) == 0

    def _maybe_distributed_barrier(self) -> None:
        torch = _import_torch()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()


class _ListPromptDataset:
    def __init__(self, items: list[Any]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Any:
        return self.items[index]


def _prompt_list_collate(batch: list[Any]) -> list[Any]:
    return batch


def _load_prompt_items_from_file(path: Path) -> list[Any]:
    if not path.exists():
        raise FileNotFoundError(f"prompt data file does not exist: {path}")

    if path.suffix == ".jsonl":
        items = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSONL record at {path}:{line_number}"
                    ) from exc
        return items

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("train", "data", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise TypeError(f"unsupported prompt file payload in {path}")


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    try:
        return int(stem.rsplit("_", 1)[-1]), stem
    except ValueError:
        return -1, stem


def _to_python_value(value: Any) -> Any:
    if isinstance(value, MappingABC):
        return {str(key): _to_python_value(item) for key, item in value.items()}
    if isinstance(value, SequenceABC) and not isinstance(value, (str, bytes, bytearray)):
        return [_to_python_value(item) for item in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _to_python_value(tolist())
        except (TypeError, ValueError):
            return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError):
            return value
    return value


def _import_torch() -> Any:
    import torch

    return torch


def _import_dataloader() -> Any:
    from torch.utils.data import DataLoader

    return DataLoader


def _import_verl_dataset_components() -> tuple[Any, Any]:
    from omegaconf import OmegaConf
    from verl.utils.agent_dataset.rl_dataset import RLHFDataset

    return OmegaConf, RLHFDataset


def _try_import_fsdp() -> Any | None:
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    except (ImportError, ModuleNotFoundError):
        return None
    return FSDP


def _get_fsdp_full_state_dict(module: Any, *, rank0_only: bool) -> Any:
    FSDP = _try_import_fsdp()
    if FSDP is None:
        return module.state_dict()
    try:
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
    except (ImportError, ModuleNotFoundError):
        return module.state_dict()
    with FSDP.state_dict_type(
        module,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=rank0_only),
    ):
        return module.state_dict()


def _load_fsdp_full_state_dict(module: Any, state_dict: Any) -> Any:
    FSDP = _try_import_fsdp()
    if FSDP is None:
        return module.load_state_dict(state_dict)
    try:
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
    except (ImportError, ModuleNotFoundError):
        return module.load_state_dict(state_dict)
    with FSDP.state_dict_type(
        module,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    ):
        return module.load_state_dict(state_dict)


def _validate_load_state_result(load_result: Any, *, module: Any) -> None:
    missing_keys = list(getattr(load_result, "missing_keys", ()))
    unexpected_keys = list(getattr(load_result, "unexpected_keys", ()))
    if not missing_keys and not unexpected_keys:
        return
    raise RuntimeError(
        "state_dict keys do not match module "
        f"{type(module).__name__}: missing_keys={missing_keys}, unexpected_keys={unexpected_keys}"
    )
