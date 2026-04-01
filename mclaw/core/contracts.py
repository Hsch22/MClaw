from __future__ import annotations

"""与外部训练栈解耦的本地协议和批数据容器。"""

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from .tree_node import TreeNode


@dataclass(slots=True)
class EnvironmentStep:
    """单步环境反馈。"""

    reward: float = 0.0
    next_state: Any | None = None
    done: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AuxiliarySample:
    """同簇 sibling 的辅助监督样本。"""

    state_tokens: list[int]
    action_tokens: list[int]
    advantage: float | None = None
    td_target: float | None = None
    cluster_id: int = -1
    cluster_weight: float = 1.0
    token_weight: float = 1.0
    source_node_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CriticSample:
    """Q-head TD 更新使用的样本。"""

    state_tokens: list[int]
    action_tokens: list[int]
    reward: float
    next_state_tokens: list[int] = field(default_factory=list)
    done: bool = False
    source_node_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrajectoryStep:
    """单个 executed action 对应的 step-level 训练记录。"""

    state_tokens: list[int] = field(default_factory=list)
    action_tokens: list[int] = field(default_factory=list)
    next_state_tokens: list[int] = field(default_factory=list)
    reward: float = 0.0
    done: bool = False
    advantage: float | None = None
    td_target: float | None = None
    state_value: float | None = None
    token_weight: float = 1.0
    source_node_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrajectoryRecord:
    """单条完整轨迹的本地表示；`responses` 仅包含 assistant action tokens。"""

    input_ids: list[int] = field(default_factory=list)
    responses: list[int] = field(default_factory=list)
    attention_mask: list[int] = field(default_factory=list)
    position_ids: list[int] = field(default_factory=list)
    response_mask: list[int] = field(default_factory=list)
    response_token_weights: list[float] = field(default_factory=list)
    advantages: list[float] = field(default_factory=list)
    returns: list[float] = field(default_factory=list)
    state_values: list[float] = field(default_factory=list)
    old_log_probs: list[float] = field(default_factory=list)
    ref_log_probs: list[float] = field(default_factory=list)
    steps: list[TrajectoryStep] = field(default_factory=list)
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


@dataclass(slots=True)
class TreeRolloutOutput:
    """一次树状 rollout 的统一输出。"""

    actor_data: ActorBatch = field(default_factory=ActorBatch)
    aux_actor_data: AuxiliaryBatch = field(default_factory=AuxiliaryBatch)
    critic_data: CriticBatch = field(default_factory=CriticBatch)
    roots: list[TreeNode] = field(default_factory=list)
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

    def step(self, action: Any, **kwargs: Any) -> tuple[Any, float, bool, Mapping[str, Any]]:
        """执行动作并返回单步环境反馈。"""


@runtime_checkable
class RolloutHandlerProtocol(Protocol):
    """轨迹拼装器的最小兼容接口。"""

    @property
    def done(self) -> bool:
        """当前分支是否已结束。"""

    @property
    def score(self) -> float:
        """当前分支累计 reward。"""

    def add_user_message(self, observation: Any, token_ids: Sequence[int] | None = None) -> None:
        """追加一轮环境 observation。"""

    def add_assistant_message(self, action: Any, token_ids: Sequence[int] | None = None) -> None:
        """追加一轮 assistant action。"""

    def record_step_advantage(self, advantage: float) -> None:
        """记录当前 step 的 advantage。"""

    def mark_done(self, done: bool) -> None:
        """更新分支结束状态。"""

    def get_generation_prompt(self, tokenizer: TokenizerProtocol) -> Sequence[int]:
        """构造下一步 vLLM generate 所需的 prompt token ids。"""

    def build_trajectory_record(self) -> TrajectoryRecord:
        """导出本地轨迹记录。"""
