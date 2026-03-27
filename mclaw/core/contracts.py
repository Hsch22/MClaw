from __future__ import annotations

"""与外部训练栈解耦的本地协议和批数据容器。"""

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from .tree_node import AuxiliarySample, CriticSample, EnvironmentStep


@dataclass(slots=True)
class TrajectoryRecord:
    """单条完整轨迹的本地表示。"""

    input_ids: list[int] = field(default_factory=list)
    attention_mask: list[int] = field(default_factory=list)
    position_ids: list[int] = field(default_factory=list)
    response_mask: list[int] = field(default_factory=list)
    advantages: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ActorBatch:
    """PPO 主样本批。"""

    trajectories: list[TrajectoryRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AuxiliaryBatch:
    """Auxiliary loss 样本批。"""

    samples: list[AuxiliarySample] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CriticBatch:
    """Q-head TD 更新样本批。"""

    samples: list[CriticSample] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class TokenizerProtocol(Protocol):
    """与外部 tokenizer 的最小兼容接口。"""

    def encode(self, text: str, add_special_tokens: bool = False, **kwargs: Any) -> list[int]:
        """将文本编码成 token ids。"""

    def decode(
        self,
        token_ids: Sequence[int],
        skip_special_tokens: bool = False,
        **kwargs: Any,
    ) -> str:
        """将 token ids 解码成文本。"""


@runtime_checkable
class InferenceEngineProtocol(Protocol):
    """与外部推理引擎的最小兼容接口。"""

    def generate(self, prompt_token_ids: Sequence[Sequence[int]], **kwargs: Any) -> Any:
        """根据 prompts 生成候选动作。"""


@runtime_checkable
class EnvironmentClientProtocol(Protocol):
    """环境实例客户端的最小兼容接口。"""

    def reset(self, item_id: Any, **kwargs: Any) -> Any:
        """重置环境到指定任务。"""

    def observe(self) -> Any:
        """读取当前环境状态。"""

    def step(self, action: Any, **kwargs: Any) -> EnvironmentStep | tuple[Any, float, bool, Mapping[str, Any]]:
        """执行动作并返回单步环境反馈。"""


@runtime_checkable
class RolloutHandlerProtocol(Protocol):
    """轨迹拼装器的最小兼容接口。"""

    def add_user_message(self, observation: Any, token_ids: Sequence[int] | None = None) -> None:
        """追加一轮环境 observation。"""

    def add_assistant_message(self, action: Any, token_ids: Sequence[int] | None = None) -> None:
        """追加一轮 assistant action。"""

    def record_step_advantage(self, advantage: float) -> None:
        """记录当前 step 的 advantage。"""

    def mark_done(self, done: bool) -> None:
        """更新分支结束状态。"""

    def build_trajectory_record(self) -> TrajectoryRecord:
        """导出本地轨迹记录。"""
