from __future__ import annotations

"""对接 agentenv / AgentGym-RL EnvClient 的适配器。"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class AgentEnvClientAdapter:
    """实现 EnvironmentClientProtocol 的环境客户端包装器。"""

    client: Any

    @classmethod
    def from_agentgym_args(cls, args: Any) -> "AgentEnvClientAdapter":
        init_env_client = _import_init_env_client()
        return cls(client=init_env_client(args))

    def reset(self, item_id: Any, **kwargs: Any) -> Any:
        """重置环境并返回 reset 后的 observation。"""
        del kwargs
        self.client.reset(item_id)
        return self.observe()

    def observe(self) -> Any:
        return self.client.observe()

    def step(self, action: Any, **kwargs: Any) -> tuple[Any, float, bool, Mapping[str, Any]]:
        """统一转成 gym 风格 `(next_state, reward, done, metadata)`。"""
        raw_step = self.client.step(action, **kwargs)
        next_state, reward, done, metadata = _coerce_env_step(raw_step)
        return next_state, reward, done, metadata


def _coerce_env_step(raw_step: Any) -> tuple[Any, float, bool, Mapping[str, Any]]:
    if isinstance(raw_step, tuple):
        if len(raw_step) != 4:
            raise ValueError(f"env step tuple must have length 4, got {len(raw_step)}")
        next_state, reward, done, metadata = raw_step
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, Mapping):
            raise TypeError("env step metadata must be a mapping")
        return next_state, float(reward), bool(done), dict(metadata)

    if isinstance(raw_step, Mapping):
        next_state = raw_step.get("state", raw_step.get("next_state"))
        reward = raw_step.get("reward", 0.0)
        done = raw_step.get("done", False)
        metadata = raw_step.get("metadata", {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, Mapping):
            raise TypeError("env step metadata must be a mapping")
        return next_state, float(reward), bool(done), dict(metadata)

    next_state = getattr(raw_step, "state", getattr(raw_step, "next_state", None))
    reward = float(getattr(raw_step, "reward", 0.0))
    done = bool(getattr(raw_step, "done", False))
    metadata = getattr(raw_step, "metadata", None)
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, Mapping):
        metadata = {}
    return next_state, reward, done, dict(metadata)


def _import_init_env_client() -> Any:
    from verl.utils.agentgym.client import init_env_client

    return init_env_client


__all__ = ["AgentEnvClientAdapter"]
