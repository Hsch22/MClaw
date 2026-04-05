from __future__ import annotations

"""对接 verl / vLLM 推理引擎的适配器。"""

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class VerlInferenceEngine:
    """实现 InferenceEngineProtocol 的 vLLM 包装器。"""

    llm: Any
    sampling_params: Any | None = None
    sampling_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sampling_params is None:
            self.sampling_params = self._build_sampling_params(self.sampling_kwargs)
        elif self.sampling_kwargs:
            raise ValueError(
                "sampling_kwargs should be empty when sampling_params is provided explicitly"
            )

    def generate(
        self,
        prompt_token_ids: Sequence[Sequence[int]],
        **kwargs: Any,
    ) -> Any:
        """对齐 verl 的 `llm.generate(...)` 调用方式。"""
        call_kwargs = dict(kwargs)
        sampling_params = call_kwargs.pop("sampling_params", self.sampling_params)
        prompts = call_kwargs.pop("prompts", None)
        use_tqdm = bool(call_kwargs.pop("use_tqdm", False))
        return self.llm.generate(
            prompts=prompts,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            use_tqdm=use_tqdm,
            **call_kwargs,
        )

    @contextmanager
    def update_sampling_params(self, **overrides: Any) -> Any:
        """临时覆写 SamplingParams 的字段。"""
        if not overrides:
            yield self.sampling_params
            return

        old_values: dict[str, Any] = {}
        for key, value in overrides.items():
            if not hasattr(self.sampling_params, key):
                continue
            old_values[key] = getattr(self.sampling_params, key)
            setattr(self.sampling_params, key, value)

        try:
            yield self.sampling_params
        finally:
            for key, value in old_values.items():
                setattr(self.sampling_params, key, value)

    def _build_sampling_params(self, kwargs: Mapping[str, Any]) -> Any:
        SamplingParams = _import_sampling_params()
        resolved_kwargs = {
            "n": 1,
            "logprobs": 1,
        }
        resolved_kwargs.update(dict(kwargs))
        if "max_tokens" not in resolved_kwargs:
            raise ValueError(
                "VerlInferenceEngine requires `sampling_kwargs['max_tokens']` unless "
                "an explicit sampling_params object is provided"
            )
        return SamplingParams(**resolved_kwargs)

    def shutdown(self) -> None:
        """显式回收 vLLM engine / worker 进程，避免脚本退出后残留。"""
        llm = getattr(self, "llm", None)
        if llm is None:
            return

        llm_engine = getattr(llm, "llm_engine", None)
        shutdown = getattr(llm_engine, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:
                pass

        model_executor = getattr(llm_engine, "model_executor", None)
        shutdown = getattr(model_executor, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:
                pass


def _import_sampling_params() -> Any:
    from vllm import SamplingParams

    return SamplingParams


__all__ = ["VerlInferenceEngine"]
