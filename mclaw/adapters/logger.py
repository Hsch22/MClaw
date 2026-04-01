from __future__ import annotations

"""对接 verl tracking 或标准 logging 的日志适配器。"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
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

    def close(self) -> None:
        close = getattr(self.tracker, "close", None)
        if callable(close):
            close()


@dataclass(slots=True)
class TensorBoardTracker:
    writer: Any

    def log(self, data: Mapping[str, float], step: int | None = None) -> None:
        for key, value in data.items():
            self.writer.add_scalar(str(key), float(value), global_step=step)
        self.writer.flush()

    def close(self) -> None:
        self.writer.close()


@dataclass(slots=True)
class WandbTracker:
    run: Any

    def log(self, data: Mapping[str, float], step: int | None = None) -> None:
        self.run.log(dict(data), step=step)

    def close(self) -> None:
        finish = getattr(self.run, "finish", None)
        if callable(finish):
            finish()


@dataclass(slots=True)
class CompositeTracker:
    trackers: list[Any]

    def log(self, data: Mapping[str, float], step: int | None = None) -> None:
        for tracker in self.trackers:
            tracker.log(data, step=step)

    def close(self) -> None:
        for tracker in self.trackers:
            close = getattr(tracker, "close", None)
            if callable(close):
                close()


def build_tracker(
    *,
    tracker_name: str,
    project_name: str,
    experiment_name: str,
    default_local_dir: str,
    path_pattern: str,
    tracker_kwargs: Mapping[str, Any] | None = None,
) -> Any | None:
    normalized_names = _normalize_tracker_names(tracker_name)
    if not normalized_names:
        return None

    trackers: list[Any] = []
    kwargs = dict(tracker_kwargs or {})
    run_name = experiment_name.strip() or "mclaw"
    root_dir = Path(default_local_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    if "tensorboard" in normalized_names:
        SummaryWriter = _import_summary_writer()
        tensorboard_dir = root_dir / "tensorboard" / run_name
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        trackers.append(TensorBoardTracker(writer=SummaryWriter(log_dir=str(tensorboard_dir))))

    if "wandb" in normalized_names:
        wandb = _import_wandb()
        wandb_dir = root_dir / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)
        trackers.append(
            WandbTracker(
                run=wandb.init(
                    project=project_name.strip() or "mclaw",
                    name=run_name,
                    dir=str(wandb_dir),
                    config={"path_pattern": path_pattern, **kwargs.pop("config", {})},
                    **kwargs,
                )
            )
        )

    if not trackers:
        return None
    if len(trackers) == 1:
        return trackers[0]
    return CompositeTracker(trackers=trackers)


def _normalize_tracker_names(tracker_name: str) -> set[str]:
    normalized = tracker_name.strip().lower()
    if not normalized or normalized == "none":
        return set()
    aliases = {
        "tb": "tensorboard",
        "tensor_board": "tensorboard",
        "both": "tensorboard,wandb",
    }
    normalized = aliases.get(normalized, normalized)
    names = {
        part.strip()
        for chunk in normalized.split("+")
        for part in chunk.split(",")
        if part.strip()
    }
    unknown = names - {"tensorboard", "wandb"}
    if unknown:
        raise ValueError(f"unsupported tracker backend(s): {sorted(unknown)}")
    return names


def _import_summary_writer() -> Any:
    from torch.utils.tensorboard import SummaryWriter

    return SummaryWriter


def _import_wandb() -> Any:
    import wandb

    return wandb


__all__ = [
    "CompositeTracker",
    "StandardLogger",
    "TensorBoardTracker",
    "WandbTracker",
    "build_tracker",
]
