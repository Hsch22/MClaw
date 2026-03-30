from __future__ import annotations

"""vLLM 兼容和特征提取辅助接口。"""

import math
import warnings
from collections.abc import Mapping as MappingABC, Sequence as SequenceABC
from contextlib import nullcontext
from typing import Any, Mapping, Sequence

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

_TOPK_LOGPROB_FIELDS = ("topk_logprobs", "token_logprobs", "logprobs")
TopKLogprobsInput = Mapping[str, Any] | Sequence[Any]


class EmbeddingMatrixCache:
    """缓存从 FSDP actor 收集得到的完整 embedding matrix。"""

    def __init__(self, actor_module_fsdp: Any) -> None:
        self.actor_module_fsdp = actor_module_fsdp
        self._embedding_matrix: torch.Tensor | None = None

    def warmup(self, refresh: bool = True) -> None:
        """在 rollout 开始前显式触发 embedding matrix materialization / all-gather。"""
        self._embedding_matrix = self.get_embedding_matrix(force_refresh=refresh)

    def get_embedding_matrix(self, force_refresh: bool = False) -> torch.Tensor:
        """返回完整 vocab 的 embedding matrix。"""
        if torch is None:
            raise ModuleNotFoundError("torch is required to access embedding matrix")
        if self._embedding_matrix is not None and not force_refresh:
            return self._embedding_matrix

        if self.actor_module_fsdp is None:
            raise ValueError("actor_module_fsdp is required to resolve embedding matrix")

        with _summon_full_params_if_needed(self.actor_module_fsdp):
            embedding_layer = _resolve_embedding_layer(self.actor_module_fsdp)
            weight = getattr(embedding_layer, "weight", None)
            if not isinstance(weight, torch.Tensor):
                raise TypeError("resolved input embedding layer must expose tensor weight")
            self._embedding_matrix = weight.detach().clone()

        if self._embedding_matrix.ndim != 2:
            raise ValueError(
                "embedding matrix must be 2D, "
                f"got shape {tuple(self._embedding_matrix.shape)}"
            )
        return self._embedding_matrix


def extract_topk_logprobs(generation_output: Any) -> Mapping[str, Any]:
    """从 vLLM 输出中抽取 top-k logprob 结构。"""
    return {
        "per_sample_topk_logprobs": _extract_per_sample_topk_logprobs(generation_output),
    }


def build_output_grad_features(
    action_token_ids: Sequence[Sequence[int]],
    topk_logprobs: TopKLogprobsInput,
    embedding_matrix: torch.Tensor,
    use_mean_pooling: bool = True,
) -> torch.Tensor:
    """把 output-gradient 近似整理为聚类特征。"""
    if torch is None:
        raise ModuleNotFoundError("torch is required to build output-grad features")

    per_sample_topk = coerce_per_sample_topk_logprobs(
        topk_logprobs,
        source_name="topk_logprobs",
    )
    if len(per_sample_topk) != len(action_token_ids):
        raise ValueError(
            "top-k logprob sample count does not match action count: "
            f"{len(per_sample_topk)} != {len(action_token_ids)}"
        )

    per_sample_features: list[torch.Tensor] = []
    max_positions = max((len(action_ids) for action_ids in action_token_ids), default=1)
    hidden_dim = int(embedding_matrix.size(-1))

    for action_ids, per_position_topk in zip(action_token_ids, per_sample_topk):
        per_position_features: list[torch.Tensor] = []
        for position, token_id in enumerate(action_ids):
            actual_embedding = embedding_matrix[int(token_id)]
            distribution = per_position_topk[position] if position < len(per_position_topk) else {}
            expected_embedding = _compute_expected_embedding(distribution, embedding_matrix)
            per_position_features.append(actual_embedding - expected_embedding)

        if not per_position_features:
            per_position_features = [torch.zeros(hidden_dim, dtype=embedding_matrix.dtype, device=embedding_matrix.device)]

        stacked = torch.stack(per_position_features, dim=0)
        if use_mean_pooling:
            per_sample_features.append(stacked.mean(dim=0))
            continue

        padded = torch.zeros(
            (max_positions, hidden_dim),
            dtype=stacked.dtype,
            device=stacked.device,
        )
        padded[: stacked.size(0)] = stacked
        per_sample_features.append(padded.reshape(-1))

    return torch.stack(per_sample_features, dim=0)


def _extract_per_sample_topk_logprobs(generation_output: Any) -> list[list[dict[int, float]]]:
    samples = _coerce_samples(generation_output)
    normalized_samples: list[list[dict[int, float]]] = []
    for sample in samples:
        topk = _resolve_topk_structure(sample)
        normalized_samples.append([_normalize_position_topk(item) for item in topk])
    return normalized_samples


def _coerce_samples(value: Any) -> list[Any]:
    if value is None:
        raise ValueError("generation_output must not be None")

    if isinstance(value, MappingABC) and "per_sample_topk_logprobs" in value:
        return coerce_per_sample_topk_logprobs(
            value,
            source_name="generation_output",
        )

    if isinstance(value, SequenceABC) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _unwrap_single_sample(item, source_name=f"generation_output[{index}]")
            for index, item in enumerate(value)
        ]

    return [_unwrap_single_sample(value, source_name="generation_output")]


def _resolve_topk_structure(sample: Any) -> list[Any]:
    for key in _TOPK_LOGPROB_FIELDS:
        if isinstance(sample, MappingABC) and key in sample:
            value = sample[key]
        elif hasattr(sample, key):
            value = getattr(sample, key)
        else:
            continue
        if isinstance(value, SequenceABC) and not isinstance(value, (str, bytes, bytearray)):
            return list(value)
        raise TypeError(
            f"{type(sample).__name__}.{key} must be a sequence of per-position top-k logprob objects"
        )

    if isinstance(sample, SequenceABC) and not isinstance(sample, (str, bytes, bytearray)):
        return list(sample)

    raise KeyError(
        f"sample of type {type(sample).__name__} must provide one of {_TOPK_LOGPROB_FIELDS}"
    )


def _normalize_position_topk(position_value: Any) -> dict[int, float]:
    if isinstance(position_value, MappingABC):
        normalized: dict[int, float] = {}
        for key, value in position_value.items():
            token_id = _coerce_token_id(key, value)
            logprob = _coerce_logprob(value)
            if token_id is None or logprob is None:
                continue
            normalized[token_id] = logprob
        if normalized or not position_value:
            return normalized
        raise TypeError("failed to parse any token/logprob pairs from mapping top-k output")

    if isinstance(position_value, SequenceABC) and not isinstance(position_value, (str, bytes, bytearray)):
        normalized: dict[int, float] = {}
        for item in position_value:
            token_id = _coerce_token_id(None, item)
            logprob = _coerce_logprob(item)
            if token_id is None or logprob is None:
                continue
            normalized[token_id] = logprob
        if normalized or not position_value:
            return normalized
        raise TypeError("failed to parse any token/logprob pairs from sequence top-k output")

    token_id = _coerce_token_id(None, position_value)
    logprob = _coerce_logprob(position_value)
    if token_id is None or logprob is None:
        raise TypeError(
            f"failed to parse token/logprob pair from top-k item of type {type(position_value).__name__}"
        )
    return {token_id: logprob}


def _coerce_token_id(key: Any, value: Any) -> int | None:
    for candidate in (key, getattr(value, "token_id", None), getattr(value, "decoded_token_id", None)):
        if candidate is None:
            continue
        try:
            return int(candidate)
        except (TypeError, ValueError):
            continue
    return None


def _coerce_logprob(value: Any) -> float | None:
    for candidate in (
        getattr(value, "logprob", None),
        value.get("logprob") if isinstance(value, MappingABC) else None,
        value,
    ):
        if candidate is None:
            continue
        try:
            scalar = float(candidate)
        except (TypeError, ValueError):
            continue
        if math.isfinite(scalar):
            return scalar
    return None


def _compute_expected_embedding(
    position_topk: Mapping[int, float],
    embedding_matrix: torch.Tensor,
) -> torch.Tensor:
    if torch is None:
        raise ModuleNotFoundError("torch is required to compute expected embeddings")
    if not position_topk:
        return torch.zeros(
            embedding_matrix.size(-1),
            dtype=embedding_matrix.dtype,
            device=embedding_matrix.device,
        )

    token_ids = torch.tensor(list(position_topk.keys()), dtype=torch.long, device=embedding_matrix.device)
    logprobs = torch.tensor(list(position_topk.values()), dtype=embedding_matrix.dtype, device=embedding_matrix.device)
    weights = torch.exp(logprobs)
    total_mass = float(weights.sum().item())
    _warn_if_low_topk_mass(total_mass, topk_size=len(position_topk))
    if not math.isfinite(total_mass) or total_mass <= 0.0:
        return torch.zeros(
            embedding_matrix.size(-1),
            dtype=embedding_matrix.dtype,
            device=embedding_matrix.device,
        )
    # 这里只能观测到 top-k token 的 logprob，因此用 top-k 子分布做归一化近似。
    # 这样 expected embedding 与 actual embedding 保持同一量级，避免特征退化成实际 token embedding 本身。
    weights = weights / total_mass
    return weights @ embedding_matrix[token_ids]


def coerce_per_sample_topk_logprobs(
    value: TopKLogprobsInput,
    source_name: str,
) -> list[Any]:
    if isinstance(value, MappingABC):
        if "per_sample_topk_logprobs" not in value:
            available_keys = ", ".join(sorted(str(key) for key in value))
            raise KeyError(
                f"{source_name} must contain 'per_sample_topk_logprobs'; "
                f"got keys: [{available_keys}]"
            )
        value = value["per_sample_topk_logprobs"]

    if isinstance(value, SequenceABC) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)

    raise TypeError(
        f"{source_name} must be either a sequence of per-sample top-k structures "
        f"or a mapping containing 'per_sample_topk_logprobs', got {type(value).__name__}"
    )


def _unwrap_single_sample(value: Any, source_name: str) -> Any:
    if _has_topk_logprob_fields(value):
        return value

    outputs = _resolve_outputs(value)
    if outputs is None:
        raise TypeError(
            f"{source_name} must be a single vLLM completion output with {_TOPK_LOGPROB_FIELDS} "
            "or a request output exposing exactly one item in 'outputs'"
        )
    if not isinstance(outputs, SequenceABC) or isinstance(outputs, (str, bytes, bytearray)):
        raise TypeError(f"{source_name}.outputs must be a sequence")

    sequence = list(outputs)
    if len(sequence) != 1:
        raise ValueError(
            f"{source_name}.outputs must contain exactly one completion output, got {len(sequence)}"
        )
    candidate = sequence[0]
    if not _has_topk_logprob_fields(candidate):
        raise KeyError(
            f"{source_name}.outputs[0] must provide one of {_TOPK_LOGPROB_FIELDS}"
        )
    return candidate


def _has_topk_logprob_fields(value: Any) -> bool:
    if isinstance(value, MappingABC) and any(field in value for field in _TOPK_LOGPROB_FIELDS):
        return True
    return any(hasattr(value, field) for field in _TOPK_LOGPROB_FIELDS)


def _resolve_outputs(value: Any) -> Any:
    if isinstance(value, MappingABC) and "outputs" in value:
        return value["outputs"]
    return getattr(value, "outputs", None)


def _resolve_embedding_layer(actor_module_fsdp: Any) -> Any:
    for candidate in (actor_module_fsdp, getattr(actor_module_fsdp, "module", None)):
        if candidate is None or not hasattr(candidate, "get_input_embeddings"):
            continue
        embedding_layer = candidate.get_input_embeddings()
        if embedding_layer is not None:
            return embedding_layer
    raise AttributeError("failed to resolve input embedding layer from actor_module_fsdp")


def _summon_full_params_if_needed(actor_module_fsdp: Any) -> Any:
    fsdp_target = _resolve_fsdp_target(actor_module_fsdp)
    if fsdp_target is None:
        return nullcontext()

    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(
            "EmbeddingMatrixCache requires torch.distributed.fsdp to gather full embedding weights"
        ) from exc

    try:
        return FSDP.summon_full_params(
            fsdp_target,
            recurse=True,
            writeback=False,
            rank0_only=False,
        )
    except (TypeError, AttributeError) as exc:
        raise RuntimeError(
            "EmbeddingMatrixCache expects the PyTorch 2.4.0 FSDP summon_full_params API "
            "(FullyShardedDataParallel.summon_full_params(module, recurse=True, "
            "writeback=False, rank0_only=False)); current runtime does not match."
        ) from exc


def _resolve_fsdp_target(actor_module_fsdp: Any) -> Any:
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    except ModuleNotFoundError:
        return None

    for candidate in (actor_module_fsdp, getattr(actor_module_fsdp, "module", None)):
        if isinstance(candidate, FSDP):
            return candidate
    return None


def _warn_if_low_topk_mass(total_mass: float, topk_size: int) -> None:
    if total_mass >= 0.5 or not _should_emit_low_topk_warning():
        return

    warnings.warn(
        "Top-k logprob mass is only %.4f for a position with %d entries; "
        "output-grad expected embedding renormalizes the observed top-k subdistribution, "
        "so small top-k may incur large approximation error." % (total_mass, topk_size),
        RuntimeWarning,
        stacklevel=2,
    )


def _should_emit_low_topk_warning() -> bool:
    if torch is None:
        return True

    distributed = getattr(torch, "distributed", None)
    if distributed is None or not hasattr(distributed, "is_available") or not distributed.is_available():
        return True
    if not hasattr(distributed, "is_initialized") or not distributed.is_initialized():
        return True
    if not hasattr(distributed, "get_rank"):
        return True
    return int(distributed.get_rank()) == 0
