from __future__ import annotations

"""树状 rollout 引擎接口。"""

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from mclaw.clustering.base import BaseClusterer, ClusterResult
from mclaw.config import TreeRolloutConfig
from mclaw.critic.q_critic import QCritic

from .branch_selector import BranchSelector
from .contracts import (
    ActorBatch,
    AuxiliaryBatch,
    CriticBatch,
    EnvironmentClientProtocol,
    InferenceEngineProtocol,
    RolloutHandlerProtocol,
    TokenizerProtocol,
)
from .tree_node import AuxiliarySample, CriticSample, TreeNode, TreeRolloutOutput


@dataclass(slots=True)
class BranchRuntime:
    """运行时分支状态。"""

    env_index: int
    current_node: TreeNode
    handler: Any | None = None
    done: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class TreeRollout:
    """树状 rollout 引擎主接口。"""

    def __init__(
        self,
        inference_engine: InferenceEngineProtocol,
        actor_module_fsdp: Any,
        q_critic: QCritic,
        clusterer: BaseClusterer,
        tokenizer: TokenizerProtocol,
        config: TreeRolloutConfig,
        env_client_factory: Callable[[], EnvironmentClientProtocol] | None = None,
        branch_selector: BranchSelector | None = None,
    ) -> None:
        self.inference_engine = inference_engine
        self.actor_module_fsdp = actor_module_fsdp
        self.q_critic = q_critic
        self.clusterer = clusterer
        self.tokenizer = tokenizer
        self.config = config
        self.env_client_factory = env_client_factory
        self.branch_selector = branch_selector or BranchSelector()

    def generate_tree_rollout(self, prompts: Any) -> TreeRolloutOutput:
        """生成树状 rollout，并返回本地 batch 容器。"""
        raise NotImplementedError("TODO: 实现树状 rollout 主循环。")

    def _initialize_env_pool(self, prompt_batch: Any) -> list[EnvironmentClientProtocol]:
        """根据输入 batch 初始化固定大小的环境实例池。"""
        raise NotImplementedError("TODO: 实现环境实例池初始化逻辑。")

    def _build_root_node(self, prompt_item: Any) -> TreeNode:
        """把单个 prompt 转成根节点。"""
        raise NotImplementedError("TODO: 实现根节点构造逻辑。")

    def _expand_root_candidates(self, root: TreeNode) -> Sequence[TreeNode]:
        """在根状态生成首轮 candidate actions。"""
        raise NotImplementedError("TODO: 实现根节点候选生成逻辑。")

    def _expand_branch_candidates(self, branches: Sequence[BranchRuntime]) -> dict[int, Sequence[TreeNode]]:
        """为所有活跃分支批量生成下一步 candidates。"""
        raise NotImplementedError("TODO: 实现活跃分支扩展逻辑。")

    def _score_candidates(self, candidates: Sequence[TreeNode]) -> Any:
        """复用 Q-critic 对候选动作做批量打分。"""
        raise NotImplementedError("TODO: 实现候选打分逻辑。")

    def _cluster_root_candidates(self, root: TreeNode, candidates: Sequence[TreeNode]) -> ClusterResult:
        """对根节点 candidates 做全局聚类。"""
        raise NotImplementedError("TODO: 实现根节点聚类逻辑。")

    def _cluster_branch_candidates(
        self,
        branch: BranchRuntime,
        candidates: Sequence[TreeNode],
    ) -> ClusterResult:
        """对单个分支的 candidates 做分支内聚类。"""
        raise NotImplementedError("TODO: 实现分支内聚类逻辑。")

    def _execute_selection(self, branch: BranchRuntime, selected_node: TreeNode) -> None:
        """在真实环境中执行选中的动作。"""
        raise NotImplementedError("TODO: 实现环境执行逻辑。")

    def _build_actor_data(self, roots: Sequence[TreeNode]) -> ActorBatch:
        """把已执行完整轨迹整理为本地 PPO 主样本。"""
        raise NotImplementedError("TODO: 实现 actor_data 构造逻辑。")

    def _build_aux_actor_data(self, aux_samples: Sequence[AuxiliarySample]) -> AuxiliaryBatch:
        """把 sibling 样本整理为本地 auxiliary loss 输入。"""
        raise NotImplementedError("TODO: 实现 aux_actor_data 构造逻辑。")

    def _build_critic_data(self, critic_samples: Sequence[CriticSample]) -> CriticBatch:
        """把 executed transitions 整理为本地 Q-head TD 更新数据。"""
        raise NotImplementedError("TODO: 实现 critic_data 构造逻辑。")
