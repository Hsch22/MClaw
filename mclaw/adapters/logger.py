from __future__ import annotations

"""对接 verl tracking 或标准 logging 的日志适配器。"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any


@dataclass(slots=True)
class StandardLogger:
    """实现 LoggerProtocol 的简单日志包装器。"""

    tracker: Any | None = None
    python_logger: logging.Logger | None = None
    level: int = logging.INFO

    def __post_init__(self) -> None:
        if self.python_logger is None:
            self.python_logger = logging.getLogger("mclaw")

    def log(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        normalized = self._normalize_metrics(metrics)
        if self.tracker is not None:
            self._log_to_tracker(normalized, step)
        if self.python_logger is not None:
            rendered = " ".join(f"{key}={value:.6f}" for key, value in sorted(normalized.items()))
            prefix = f"step={step} " if step is not None else ""
            self.python_logger.log(self.level, "%s%s", prefix, rendered)

    def _log_to_tracker(self, metrics: Mapping[str, float], step: int | None) -> None:
        log_fn = getattr(self.tracker, "log", None)
        if not callable(log_fn):
            raise TypeError("tracker must expose a callable log(...) method")

        try:
            log_fn(data=dict(metrics), step=step)
            return
        except TypeError:
            pass

        try:
            if step is None:
                log_fn(dict(metrics))
            else:
                log_fn(dict(metrics), step)
            return
        except TypeError as exc:
            raise TypeError(
                "tracker.log must accept either (data=..., step=...) or positional "
                "(metrics, step)"
            ) from exc

    def _normalize_metrics(self, metrics: Mapping[str, Any]) -> dict[str, float]:
        normalized: dict[str, float] = {}
        for key, value in metrics.items():
            metric_value = self._to_float(value)
            if metric_value is None:
                continue
            normalized[str(key)] = metric_value
        return normalized

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


__all__ = ["StandardLogger"]
