from __future__ import annotations

"""对接 verl DataParallelPPOActor 的 actor backend。"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Any

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

from mclaw.core import ActorBatch, AuxiliaryBatch, AuxiliarySample

from .dataproto_adapter import AdaptedActorBatch, DataProtoAdapter


@dataclass(slots=True)
class VerlActorBackend:
    """实现 ActorBackendProtocol 的 verl 适配器。"""

    actor: Any
    adapter: DataProtoAdapter = field(default_factory=DataProtoAdapter)
    dataproto_meta_info: dict[str, Any] = field(default_factory=dict)
    use_kl_loss: bool | None = None
    aux_loss_config: dict[str, Any] = field(default_factory=dict)

    def compute_log_prob(self, batch: Any) -> Mapping[str, Any]:
        """重算 old log-probs，并扩展回 full-sequence 形状。"""
        source_batch, data = self._resolve_batch_and_dataproto(
            batch,
            include_ref_log_prob=False,
        )
        data = self._move_dataproto_to_actor_device(data)
        raw_output = self.actor.compute_log_prob(data)
        payload = self.adapter.unwrap_signal_output(
            raw_output,
            preferred_keys=("old_log_probs", "log_probs"),
        )
        if source_batch is None:
            return {"old_log_probs": payload}
        full_sequence_values = self.adapter.apply_signal_to_batch(
            source_batch,
            field_name="old_log_probs",
            payload=payload,
        )
        return {"old_log_probs": full_sequence_values}

    def update_policy(self, batch: Any) -> Mapping[str, float]:
        """执行 PPO update，并把后端 list-metrics 规约成标量。"""
        source_batch, data = self._resolve_batch_and_dataproto(
            batch,
            include_ref_log_prob=self._should_include_ref_log_prob(),
        )
        data = self._move_dataproto_to_actor_device(data)
        aux_batch = self._resolve_auxiliary_batch(source_batch)
        if aux_batch is not None and aux_batch.samples:
            return self._update_policy_with_auxiliary(data=data, aux_batch=aux_batch)
        raw_output = self.actor.update_policy(data)
        metrics = self._extract_metrics_mapping(raw_output)
        return self._reduce_metrics(metrics)

    def update_aux_loss(self, batch: AuxiliaryBatch) -> Mapping[str, float]:
        """在 MClaw 侧直接计算 auxiliary policy gradient。"""
        if torch is None or F is None:
            raise ModuleNotFoundError("torch is required to update auxiliary loss")

        samples = self._resolve_auxiliary_samples(batch)
        coef = float(self.aux_loss_config.get("coef", 0.0))
        if coef == 0.0 or not samples:
            return {
                "actor/aux_loss": 0.0,
                "actor/aux_samples": float(len(samples)),
                "actor/aux_coef": coef,
            }

        actor_module = getattr(self.actor, "actor_module", None)
        actor_optimizer = getattr(self.actor, "actor_optimizer", None)
        if actor_module is None or actor_optimizer is None:
            return {}

        actor_module.train()
        device = _resolve_module_device(actor_module)
        if device is None:
            return {}

        valid_samples = [sample for sample in samples if sample.action_tokens]
        if not valid_samples:
            return {
                "actor/aux_loss": 0.0,
                "actor/aux_samples": 0.0,
                "actor/aux_coef": coef,
            }

        sample_weights = [self._resolve_auxiliary_sample_weight(sample) for sample in valid_samples]
        abs_weight_sum = sum(abs(weight) for weight in sample_weights)
        if abs_weight_sum == 0.0:
            return {
                "actor/aux_loss": 0.0,
                "actor/aux_samples": float(len(valid_samples)),
                "actor/aux_coef": coef,
                "actor/aux_weight_abs_sum": 0.0,
            }

        micro_batch_size = max(int(self.dataproto_meta_info.get("micro_batch_size", len(valid_samples))), 1)
        actor_optimizer.zero_grad()

        total_aux_loss = 0.0
        total_mean_log_prob = 0.0
        total_effective_tokens = 0.0
        n_micro_batches = 0

        for start in range(0, len(valid_samples), micro_batch_size):
            chunk = valid_samples[start : start + micro_batch_size]
            chunk_weights = sample_weights[start : start + micro_batch_size]
            micro_loss, micro_metrics = self._compute_auxiliary_micro_batch_loss(
                actor_module=actor_module,
                device=device,
                samples=chunk,
                sample_weights=chunk_weights,
                coef=coef,
            )
            self._backward_loss(micro_loss)
            total_aux_loss += float(micro_loss.detach().item())
            total_mean_log_prob += micro_metrics["mean_log_prob"]
            total_effective_tokens += micro_metrics["effective_tokens"]
            n_micro_batches += 1

        self._optimizer_step()

        mean_log_prob = total_mean_log_prob / max(n_micro_batches, 1)
        return {
            "actor/aux_loss": total_aux_loss,
            "actor/aux_samples": float(len(valid_samples)),
            "actor/aux_coef": coef,
            "actor/aux_mean_log_prob": mean_log_prob,
            "actor/aux_effective_tokens": total_effective_tokens,
            "actor/aux_weight_abs_sum": abs_weight_sum,
        }

    def _compute_auxiliary_micro_batch_loss(
        self,
        *,
        actor_module: Any,
        device: Any,
        samples: Sequence[AuxiliarySample],
        sample_weights: Sequence[float],
        coef: float,
    ) -> tuple[Any, dict[str, float]]:
        if torch is None or F is None:
            raise ModuleNotFoundError("torch is required to compute auxiliary loss")

        pad_token_id = int(self.adapter.pad_token_id)
        sequences: list[list[int]] = []
        attention_rows: list[list[int]] = []
        action_loss_masks: list[list[float]] = []
        token_scales: list[float] = []

        for sample in samples:
            state_tokens = [int(token_id) for token_id in sample.state_tokens]
            if not state_tokens:
                state_tokens = [pad_token_id]
            action_tokens = [int(token_id) for token_id in sample.action_tokens]
            sequence = state_tokens + action_tokens
            sequences.append(sequence)
            attention_rows.append([1] * len(sequence))

            action_mask = [0.0] * max(len(sequence) - 1, 0)
            start_index = max(len(state_tokens) - 1, 0)
            end_index = min(start_index + len(action_tokens), len(action_mask))
            for token_index in range(start_index, end_index):
                action_mask[token_index] = 1.0
            action_loss_masks.append(action_mask)

            token_scale = float(sample.token_weight) if sample.token_weight else 1.0 / max(
                len(action_tokens),
                1,
            )
            token_scales.append(token_scale)

        input_ids = _pad_int_sequences(sequences, pad_value=pad_token_id, device=device)
        attention_mask = _pad_int_sequences(attention_rows, pad_value=0, device=device)
        shifted_mask = _pad_float_sequences(action_loss_masks, pad_value=0.0, device=device)
        sample_weight_tensor = torch.tensor(sample_weights, dtype=torch.float32, device=device)
        token_scale_tensor = torch.tensor(token_scales, dtype=torch.float32, device=device)

        outputs = actor_module(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        logits = getattr(outputs, "logits", None)
        if logits is None:
            raise ValueError("actor_module forward must return logits to compute auxiliary loss")

        shifted_logits = logits[:, :-1, :].float()
        shifted_targets = input_ids[:, 1:]
        token_log_probs = F.log_softmax(shifted_logits, dim=-1).gather(
            dim=-1,
            index=shifted_targets.unsqueeze(-1),
        ).squeeze(-1)

        sequence_log_probs = (token_log_probs * shifted_mask).sum(dim=-1) * token_scale_tensor
        micro_loss = -coef * torch.sum(sample_weight_tensor * sequence_log_probs)

        return micro_loss, {
            "mean_log_prob": float(sequence_log_probs.detach().mean().item()),
            "effective_tokens": float(shifted_mask.detach().sum().item()),
        }

    def _update_policy_with_auxiliary(
        self,
        *,
        data: Any,
        aux_batch: AuxiliaryBatch,
    ) -> Mapping[str, float]:
        if torch is None:
            raise ModuleNotFoundError("torch is required to run joint PPO + auxiliary updates")

        agg_loss, append_to_dict, get_device_id, get_policy_loss_fn, kl_penalty, prepare_dynamic_batch = (
            _import_verl_policy_update_helpers()
        )

        actor_config = getattr(self.actor, "config", None)
        actor_module = getattr(self.actor, "actor_module", None)
        actor_optimizer = getattr(self.actor, "actor_optimizer", None)
        if actor_config is None or actor_module is None or actor_optimizer is None:
            raw_output = self.actor.update_policy(data)
            return self._reduce_metrics(self._extract_metrics_mapping(raw_output))

        actor_module.train()
        temperature = data.meta_info["temperature"]
        pad_token_id = data.meta_info.get("pad_token_id", 0)

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if getattr(self.actor, "use_prefix_grouper", False) and "prompts" in data.batch.keys():
            select_keys.append("prompts")
        if self._should_include_ref_log_prob():
            select_keys.append("ref_log_prob")
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")

        non_tensor_select_keys: list[str] = []
        if "multi_modal_inputs" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("multi_modal_inputs")
        if getattr(self.actor, "use_prefix_grouper", False) and "uid" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("uid")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)
        mini_batches = data.split(int(actor_config.ppo_mini_batch_size))
        on_policy = len(mini_batches) == 1 and int(getattr(actor_config, "ppo_epochs", 1)) == 1
        n_mini_batches = max(len(mini_batches), 1)

        metrics: dict[str, Any] = {
            "actor/pg_loss": 0.0,
            "actor/kl_loss": 0.0,
            "actor/aux_loss": 0.0,
        }

        for _ in range(max(int(getattr(actor_config, "ppo_epochs", 1)), 1)):
            for mini_batch in mini_batches:
                if bool(getattr(actor_config, "use_dynamic_bsz", False)):
                    max_token_len = (
                        int(getattr(actor_config, "ppo_max_token_len_per_gpu"))
                        * int(getattr(self.actor, "ulysses_sequence_parallel_size", 1))
                    )
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                    gradient_accumulation = 1
                else:
                    gradient_accumulation = max(
                        int(actor_config.ppo_mini_batch_size)
                        // int(actor_config.ppo_micro_batch_size_per_gpu),
                        1,
                    )
                    micro_batches = mini_batch.split(int(actor_config.ppo_micro_batch_size_per_gpu))

                actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics: dict[str, Any] = {}
                    model_inputs = {
                        **micro_batch.batch,
                        **micro_batch.non_tensor_batch,
                        "pad_token_id": pad_token_id,
                    }
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = float(getattr(actor_config, "entropy_coeff", 0.0))
                    loss_agg_mode = str(getattr(actor_config, "loss_agg_mode", "token-mean"))
                    calculate_entropy = bool(getattr(actor_config, "calculate_entropy", False)) or (
                        entropy_coeff != 0
                    )
                    if bool(getattr(actor_config, "use_dynamic_bsz", False)):
                        loss_scale_factor = response_mask.shape[0] / int(actor_config.ppo_mini_batch_size)
                    else:
                        loss_scale_factor = 1.0 / float(gradient_accumulation)

                    outputs = self.actor._forward_micro_batch(
                        model_inputs,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
                    )
                    log_prob = outputs["log_probs"]
                    entropy = outputs["entropys"] if calculate_entropy else None

                    if getattr(actor_config, "use_rollout_log_probs", False):
                        old_log_prob = model_inputs["old_log_probs"]
                    elif on_policy:
                        old_log_prob = log_prob.detach()

                    policy_loss_fn = get_policy_loss_fn(
                        _resolve_nested_value(actor_config, ("policy_loss", "loss_mode"), default="vanilla")
                    )
                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=actor_config,
                        rollout_is_weights=model_inputs.get("rollout_is_weights"),
                    )
                    micro_batch_metrics.update(pg_metrics)

                    policy_loss = pg_loss
                    if calculate_entropy and entropy is not None:
                        entropy_agg = agg_loss(
                            loss_mat=entropy,
                            loss_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                        )
                        micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                        if entropy_coeff != 0:
                            policy_loss -= entropy_agg * entropy_coeff

                    if self._should_include_ref_log_prob():
                        ref_log_prob = model_inputs["ref_log_prob"]
                        kld = kl_penalty(
                            logprob=log_prob,
                            ref_logprob=ref_log_prob,
                            kl_penalty=str(getattr(actor_config, "kl_loss_type", "kl")),
                        )
                        kl_loss = agg_loss(
                            loss_mat=kld,
                            loss_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                        )
                        policy_loss = policy_loss + kl_loss * float(getattr(actor_config, "kl_loss_coef", 0.0))
                        metrics["actor/kl_loss"] += kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = float(
                            getattr(actor_config, "kl_loss_coef", 0.0)
                        )

                    self._backward_loss(policy_loss * loss_scale_factor)
                    metrics["actor/pg_loss"] += pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                aux_metrics = self._backward_auxiliary_loss(
                    aux_batch=aux_batch,
                    global_scale=1.0 / float(n_mini_batches),
                )
                self._merge_aux_metrics(metrics, aux_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(
                    metrics,
                    {"actor/grad_norm": float(grad_norm.detach().item()) if hasattr(grad_norm, "detach") else float(grad_norm)},
                )

        actor_optimizer.zero_grad()
        return self._reduce_metrics(metrics)

    def _backward_auxiliary_loss(
        self,
        *,
        aux_batch: AuxiliaryBatch,
        global_scale: float,
    ) -> Mapping[str, float]:
        if torch is None or F is None:
            raise ModuleNotFoundError("torch is required to backprop auxiliary loss")

        actor_module = getattr(self.actor, "actor_module", None)
        if actor_module is None:
            return {}

        device = _resolve_module_device(actor_module)
        if device is None:
            return {}

        samples = [sample for sample in self._resolve_auxiliary_samples(aux_batch) if sample.action_tokens]
        coef = float(self.aux_loss_config.get("coef", 0.0))
        if coef == 0.0 or not samples or global_scale == 0.0:
            return {
                "actor/aux_loss": 0.0,
                "actor/aux_samples": float(len(samples)),
                "actor/aux_coef": coef,
                "actor/aux_weight_abs_sum": sum(
                    abs(self._resolve_auxiliary_sample_weight(sample)) for sample in samples
                ),
            }

        sample_weights = [self._resolve_auxiliary_sample_weight(sample) for sample in samples]
        abs_weight_sum = sum(abs(weight) for weight in sample_weights)
        if abs_weight_sum == 0.0:
            return {
                "actor/aux_loss": 0.0,
                "actor/aux_samples": float(len(samples)),
                "actor/aux_coef": coef,
                "actor/aux_weight_abs_sum": 0.0,
            }

        micro_batch_size = max(int(self.dataproto_meta_info.get("micro_batch_size", len(samples))), 1)
        total_aux_loss = 0.0
        total_mean_log_prob = 0.0
        total_effective_tokens = 0.0
        n_micro_batches = 0

        for start in range(0, len(samples), micro_batch_size):
            chunk = samples[start : start + micro_batch_size]
            chunk_weights = sample_weights[start : start + micro_batch_size]
            micro_loss, micro_metrics = self._compute_auxiliary_micro_batch_loss(
                actor_module=actor_module,
                device=device,
                samples=chunk,
                sample_weights=chunk_weights,
                coef=coef,
            )
            self._backward_loss(micro_loss * float(global_scale))
            total_aux_loss += float(micro_loss.detach().item()) * float(global_scale)
            total_mean_log_prob += micro_metrics["mean_log_prob"]
            total_effective_tokens += micro_metrics["effective_tokens"]
            n_micro_batches += 1

        mean_log_prob = total_mean_log_prob / max(n_micro_batches, 1)
        return {
            "actor/aux_loss": total_aux_loss,
            "actor/aux_samples": float(len(samples)),
            "actor/aux_coef": coef,
            "actor/aux_mean_log_prob": mean_log_prob,
            "actor/aux_effective_tokens": total_effective_tokens,
            "actor/aux_weight_abs_sum": abs_weight_sum,
        }

    def _resolve_auxiliary_samples(self, batch: Any) -> list[AuxiliarySample]:
        if isinstance(batch, AuxiliaryBatch):
            return list(batch.samples)
        if isinstance(batch, Sequence) and not isinstance(batch, (str, bytes, bytearray)):
            return [sample for sample in batch if isinstance(sample, AuxiliarySample)]
        return []

    def _resolve_auxiliary_batch(self, source_batch: ActorBatch | None) -> AuxiliaryBatch | None:
        if source_batch is None:
            return None
        aux_batch = source_batch.metadata.get("auxiliary_batch")
        if isinstance(aux_batch, AuxiliaryBatch):
            return aux_batch
        aux_samples = source_batch.metadata.get("auxiliary_samples")
        if isinstance(aux_samples, Sequence) and not isinstance(aux_samples, (str, bytes, bytearray)):
            return AuxiliaryBatch(samples=[sample for sample in aux_samples if isinstance(sample, AuxiliarySample)])
        return None

    def _merge_aux_metrics(self, metrics: dict[str, Any], aux_metrics: Mapping[str, float]) -> None:
        for key, value in aux_metrics.items():
            if key in {"actor/aux_loss"}:
                metrics[key] = float(metrics.get(key, 0.0)) + float(value)
                continue
            if key in {"actor/aux_samples", "actor/aux_coef", "actor/aux_weight_abs_sum"}:
                metrics[key] = float(value)
                continue
            bucket = metrics.get(key)
            if not isinstance(bucket, list):
                bucket = []
                metrics[key] = bucket
            bucket.append(float(value))

    def _resolve_auxiliary_sample_weight(self, sample: AuxiliarySample) -> float:
        signal = 1.0
        if bool(self.aux_loss_config.get("use_same_advantage", True)):
            if sample.advantage is not None:
                signal = float(sample.advantage)
        elif sample.td_target is not None:
            signal = float(sample.td_target)
        elif sample.advantage is not None:
            signal = float(sample.advantage)
        return float(sample.cluster_weight) * signal

    def _backward_loss(self, loss: Any) -> None:
        scaler = getattr(self.actor, "scaler", None)
        if scaler is not None:
            scaler.scale(loss).backward()
            return
        loss.backward()

    def _optimizer_step(self) -> Any:
        optimizer_step = getattr(self.actor, "_optimizer_step", None)
        if callable(optimizer_step):
            return optimizer_step()
        optimizer = getattr(self.actor, "actor_optimizer", None)
        if optimizer is None:
            raise ValueError("actor optimizer is required for policy updates")
        optimizer.step()
        if torch is not None:
            return torch.tensor(0.0)
        return 0.0

    def _should_include_ref_log_prob(self) -> bool:
        if self.use_kl_loss is not None:
            return bool(self.use_kl_loss)
        return bool(_resolve_nested_value(self.actor, ("config", "use_kl_loss"), default=False))

    def _resolve_batch_and_dataproto(
        self,
        batch: Any,
        *,
        include_ref_log_prob: bool,
    ) -> tuple[ActorBatch | None, Any]:
        if isinstance(batch, AdaptedActorBatch):
            return batch.source_batch, batch.dataproto

        if isinstance(batch, ActorBatch):
            data = self.adapter.to_dataproto(
                batch,
                include_ref_log_prob=include_ref_log_prob,
                meta_info_overrides=self.dataproto_meta_info,
            )
            return batch, data

        if hasattr(batch, "batch") and hasattr(batch, "meta_info"):
            return None, batch

        raise TypeError(
            "batch must be an ActorBatch, an AdaptedActorBatch, or a DataProto-like object"
        )

    def _move_dataproto_to_actor_device(self, data: Any) -> Any:
        to_device = getattr(data, "to", None)
        if not callable(to_device):
            return data

        actor_module = getattr(self.actor, "actor_module", None)
        device = _resolve_module_device(actor_module)
        if device is None:
            return data
        return data.to(device)

    def _extract_metrics_mapping(self, output: Any) -> Mapping[str, Any]:
        if output is None:
            return {}

        if isinstance(output, Mapping):
            return output

        meta_info = getattr(output, "meta_info", None)
        if isinstance(meta_info, Mapping):
            metrics = meta_info.get("metrics")
            if isinstance(metrics, Mapping):
                return metrics
            return {}

        raise TypeError(
            "actor.update_policy() must return a metrics mapping or a DataProto-like "
            "object with meta_info['metrics']"
        )

    def _reduce_metrics(self, metrics: Mapping[str, Any]) -> dict[str, float]:
        reduced: dict[str, float] = {}
        for key, value in metrics.items():
            metric_value = self._reduce_metric_value(value)
            if metric_value is None:
                continue
            reduced[key] = metric_value
        return reduced

    def _reduce_metric_value(self, value: Any) -> float | None:
        scalar = self._to_float(value)
        if scalar is not None:
            return scalar

        sequence = self._to_sequence(value)
        if sequence is None or not sequence:
            return None

        total = 0.0
        count = 0
        for item in sequence:
            item_scalar = self._to_float(item)
            if item_scalar is None:
                continue
            total += item_scalar
            count += 1
        if count == 0:
            return None
        return total / count

    def _to_float(self, value: Any) -> float | None:
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

    def _to_sequence(self, value: Any) -> list[Any] | None:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return list(value)
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            try:
                converted = tolist()
            except (TypeError, ValueError):
                return None
            if isinstance(converted, Sequence) and not isinstance(
                converted,
                (str, bytes, bytearray),
            ):
                return list(converted)
        return None


def _resolve_nested_value(value: Any, path: tuple[str, ...], default: Any) -> Any:
    current = value
    for key in path:
        if current is None:
            return default
        if isinstance(current, Mapping):
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


def _resolve_module_device(module: Any) -> Any | None:
    if module is None:
        return None
    try:
        parameter = next(module.parameters())
    except (StopIteration, AttributeError, TypeError):
        return None
    return parameter.device


def _import_verl_policy_update_helpers() -> tuple[Any, Any, Any, Any, Any, Any]:
    from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
    from verl.utils.device import get_device_id
    from verl.utils.py_functional import append_to_dict
    from verl.utils.seqlen_balancing import prepare_dynamic_batch

    return agg_loss, append_to_dict, get_device_id, get_policy_loss_fn, kl_penalty, prepare_dynamic_batch


def _pad_int_sequences(
    rows: Sequence[Sequence[int]],
    *,
    pad_value: int,
    device: Any,
) -> Any:
    if torch is None:
        raise ModuleNotFoundError("torch is required to pad auxiliary sequences")
    max_length = max((len(row) for row in rows), default=0)
    padded = [
        [int(value) for value in row] + ([int(pad_value)] * (max_length - len(row)))
        for row in rows
    ]
    return torch.tensor(padded, dtype=torch.long, device=device)


def _pad_float_sequences(
    rows: Sequence[Sequence[float]],
    *,
    pad_value: float,
    device: Any,
) -> Any:
    if torch is None:
        raise ModuleNotFoundError("torch is required to pad auxiliary sequences")
    max_length = max((len(row) for row in rows), default=0)
    padded = [
        [float(value) for value in row] + ([float(pad_value)] * (max_length - len(row)))
        for row in rows
    ]
    return torch.tensor(padded, dtype=torch.float32, device=device)


__all__ = ["VerlActorBackend"]
