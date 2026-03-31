from __future__ import annotations

"""树状 rollout 引擎接口。"""

from collections.abc import Mapping, Sequence as SequenceABC
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from mclaw.clustering.base import BaseClusterer, ClusterResult
from mclaw.config import TreeRolloutConfig
from mclaw.critic import compute_tree_advantage, estimate_state_value
from mclaw.critic.q_critic import QCritic, QCriticOutput

from .branch_selector import BranchSelector, SelectionResult
from .contracts import (
    ActorBatch,
    AuxiliarySample,
    AuxiliaryBatch,
    CriticSample,
    CriticBatch,
    EnvironmentStep,
    EnvironmentClientProtocol,
    InferenceEngineProtocol,
    RolloutHandlerProtocol,
    TreeRolloutOutput,
    TokenizerProtocol,
    TrajectoryRecord,
    TrajectoryStep,
)
from .tree_node import TreeNode, resolve_action_token_weight


@dataclass(slots=True)
class BranchRuntime:
    """运行时分支状态。"""

    env_index: int
    current_node: TreeNode
    handler: Any | None = None
    done: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CandidateRequest:
    """一次候选动作展开请求。"""

    owner_id: Any
    parent: TreeNode
    state_tokens: list[int]
    budget: int


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
        self._node_counter = 0

    def generate_tree_rollout(self, prompts: Any) -> TreeRolloutOutput:
        """生成树状 rollout，并返回本地 batch 容器。"""
        roots: list[TreeNode] = []

        for prompt_item in self._normalize_prompt_items(prompts):
            env_pool = self._initialize_env_pool(prompt_item)
            root = self._build_root_node(prompt_item)
            roots.append(root)

            root_candidates = list(self._expand_root_candidates(root))
            if not root_candidates:
                continue

            root_cluster = self._cluster_root_candidates(root, root_candidates)
            n_select = min(
                self.config.n_envs,
                len(root_candidates),
                len(env_pool) if env_pool else self.config.n_envs,
            )
            selected_root_indices = self.branch_selector.select_root_representatives(
                candidates=root_candidates,
                representative_indices=root_cluster.representative_indices,
                cluster_labels=root_cluster.labels,
                n_select=n_select,
            )
            root.metadata["cluster_labels"] = list(root_cluster.labels)
            root.metadata["representative_indices"] = list(root_cluster.representative_indices)
            root.metadata["selected_indices"] = list(selected_root_indices)

            active_branches: list[BranchRuntime] = []
            for branch_slot, candidate_index in enumerate(selected_root_indices):
                branch = BranchRuntime(
                    env_index=branch_slot,
                    current_node=root_candidates[candidate_index],
                    done=False,
                    metadata={
                        "prompt_item": prompt_item,
                    },
                )
                if env_pool and branch_slot < len(env_pool):
                    branch.metadata["env_client"] = env_pool[branch_slot]
                branch.current_node.branch_id = branch_slot
                self._execute_selection(branch, branch.current_node)
                active_branches.append(branch)

            for _ in range(1, self.config.max_rounds):
                live_branches = [branch for branch in active_branches if not branch.done]
                if not live_branches:
                    break

                candidates_by_branch = self._expand_branch_candidates(live_branches)
                for branch in live_branches:
                    candidates = list(candidates_by_branch.get(branch.env_index, ()))
                    if not candidates:
                        branch.done = True
                        continue

                    cluster_result = self._cluster_branch_candidates(branch, candidates)
                    selection = self.branch_selector.select_branch_action(
                        candidates=candidates,
                        representative_indices=cluster_result.representative_indices,
                        cluster_labels=cluster_result.labels,
                    )
                    branch.current_node.metadata["selection"] = selection
                    branch.current_node.metadata["cluster_labels"] = list(cluster_result.labels)
                    branch.current_node.metadata["representative_indices"] = list(
                        cluster_result.representative_indices
                    )

                    selected_node = candidates[selection.selected_index]
                    selected_node.branch_id = branch.env_index
                    self._execute_selection(branch, selected_node)

        gamma = float(getattr(self.q_critic.config, "gamma", 0.99))
        for root in roots:
            compute_tree_advantage([root], gamma=gamma)

        aux_samples = self._collect_auxiliary_samples(roots)
        critic_samples = self._collect_critic_samples(roots)

        actor_data = self._build_actor_data(roots)
        aux_actor_data = self._build_aux_actor_data(aux_samples)
        critic_data = self._build_critic_data(critic_samples)
        return TreeRolloutOutput(
            actor_data=actor_data,
            aux_actor_data=aux_actor_data,
            critic_data=critic_data,
            roots=roots,
            metadata={
                "n_roots": len(roots),
                "n_aux_samples": len(aux_samples),
                "n_critic_samples": len(critic_samples),
            },
        )

    def _initialize_env_pool(self, prompt_batch: Any) -> list[EnvironmentClientProtocol]:
        """根据输入 batch 初始化固定大小的环境实例池。"""
        if self.env_client_factory is None:
            return []

        env_pool = [self.env_client_factory() for _ in range(self.config.n_envs)]
        item_id = _resolve_field(prompt_batch, "item_id")
        reset_kwargs = _resolve_mapping(prompt_batch, "env_reset_kwargs") or {}
        for env_client in env_pool:
            if hasattr(env_client, "reset") and item_id is not None:
                env_client.reset(item_id, **reset_kwargs)
        return env_pool

    def _build_root_node(self, prompt_item: Any) -> TreeNode:
        """把单个 prompt 转成根节点。"""
        if isinstance(prompt_item, TreeNode):
            return prompt_item

        state_tokens = _resolve_token_ids(prompt_item, tokenizer=self.tokenizer)
        state_text = _resolve_text(prompt_item)
        if not state_text and state_tokens:
            state_text = self.tokenizer.decode(state_tokens, skip_special_tokens=False)

        node_id = str(_resolve_field(prompt_item, "node_id") or self._next_node_id("root"))
        return TreeNode(
            state_tokens=state_tokens,
            state_text=state_text,
            depth=0,
            node_id=node_id,
            metadata={
                "prompt_item": prompt_item,
                "item_id": _resolve_field(prompt_item, "item_id"),
            },
        )

    def _expand_root_candidates(self, root: TreeNode) -> Sequence[TreeNode]:
        """在根状态生成首轮 candidate actions。"""
        requests = [
            CandidateRequest(
                owner_id=root.node_id,
                parent=root,
                state_tokens=list(root.state_tokens),
                budget=self.config.root_budget,
            )
        ]
        candidates_by_owner = self._generate_candidates_for_requests(requests)
        candidates = list(candidates_by_owner.get(root.node_id, ()))
        self._score_candidate_groups([(root, candidates)])
        return candidates

    def _expand_branch_candidates(self, branches: Sequence[BranchRuntime]) -> dict[int, Sequence[TreeNode]]:
        """为所有活跃分支批量生成下一步 candidates。"""
        requests: list[CandidateRequest] = []
        for branch in branches:
            if branch.done:
                continue
            parent = branch.current_node
            state_tokens = parent.next_state_tokens or parent.state_tokens
            requests.append(
                CandidateRequest(
                    owner_id=branch.env_index,
                    parent=parent,
                    state_tokens=list(state_tokens),
                    budget=self.config.branch_budget,
                )
            )
        candidates_by_owner = self._generate_candidates_for_requests(requests)
        self._score_candidate_groups(
            [
                (request.parent, list(candidates_by_owner.get(request.owner_id, ())))
                for request in requests
            ]
        )
        return {
            int(request.owner_id): list(candidates_by_owner.get(request.owner_id, ()))
            for request in requests
        }

    def _score_candidates(self, candidates: Sequence[TreeNode]) -> QCriticOutput:
        """复用 Q-critic 对候选动作做批量打分。"""
        if not candidates:
            return QCriticOutput()

        first_state_tokens = list(candidates[0].state_tokens)
        shared_state = all(list(node.state_tokens) == first_state_tokens for node in candidates)
        state_token_ids: Sequence[int] | Sequence[Sequence[int]]
        if shared_state:
            state_token_ids = first_state_tokens
        else:
            state_token_ids = [list(node.state_tokens) for node in candidates]

        return self.q_critic.score_actions(
            state_token_ids=state_token_ids,
            action_token_ids=[node.action_tokens for node in candidates],
        )

    def _score_candidate_groups(
        self,
        candidate_groups: Sequence[tuple[TreeNode, Sequence[TreeNode]]],
    ) -> None:
        normalized_groups = [
            (parent, list(candidates))
            for parent, candidates in candidate_groups
            if candidates
        ]
        if not normalized_groups:
            return

        flat_candidates: list[TreeNode] = []
        parent_slices: list[tuple[TreeNode, int, int]] = []
        for parent, candidates in normalized_groups:
            start = len(flat_candidates)
            flat_candidates.extend(candidates)
            parent_slices.append((parent, start, len(flat_candidates)))

        qcritic_output = self._score_candidates(flat_candidates)
        _merge_qcritic_outputs_into_nodes(flat_candidates, qcritic_output)

        total_candidates = len(flat_candidates)
        for parent, start, end in parent_slices:
            model_outputs: dict[str, Any] = {}
            generation_output = parent.metadata.get("generation_output")
            if generation_output is not None:
                model_outputs["generation_output"] = generation_output
            model_outputs.update(
                _qcritic_output_to_mapping(
                    _slice_qcritic_output(qcritic_output, start=start, end=end, total=total_candidates)
                )
            )
            parent.metadata["candidate_model_outputs"] = model_outputs

    def _cluster_root_candidates(self, root: TreeNode, candidates: Sequence[TreeNode]) -> ClusterResult:
        """对根节点 candidates 做全局聚类。"""
        model_outputs = root.metadata.get("candidate_model_outputs", {})
        return self.clusterer.cluster_candidates(
            nodes=candidates,
            n_clusters=self.config.root_clusters,
            model_outputs=model_outputs,
        )

    def _cluster_branch_candidates(
        self,
        branch: BranchRuntime,
        candidates: Sequence[TreeNode],
    ) -> ClusterResult:
        """对单个分支的 candidates 做分支内聚类。"""
        model_outputs = branch.current_node.metadata.get("candidate_model_outputs", {})
        return self.clusterer.cluster_candidates(
            nodes=candidates,
            n_clusters=self.config.intra_branch_clusters,
            model_outputs=model_outputs,
        )

    def _execute_selection(self, branch: BranchRuntime, selected_node: TreeNode) -> None:
        """在真实环境中执行选中的动作。"""
        env_step = self._run_environment_step(branch, selected_node)
        selected_node.executed = True
        selected_node.selected_for_execution = True
        selected_node.env_reward = float(env_step.reward)
        selected_node.env_next_state = env_step.next_state
        selected_node.done = bool(env_step.done)

        observation_text = _to_text(env_step.next_state)
        observation_tokens = _encode_text(self.tokenizer, observation_text) if observation_text else []
        selected_node.metadata["observation_tokens"] = observation_tokens
        selected_node.metadata["observation_text"] = observation_text
        selected_node.metadata["env_step_metadata"] = dict(env_step.metadata)
        selected_node.next_state_tokens = (
            list(selected_node.state_tokens)
            + list(selected_node.action_tokens)
            + observation_tokens
        )

        if branch.handler is not None:
            if hasattr(branch.handler, "add_assistant_message"):
                branch.handler.add_assistant_message(
                    selected_node.action_text or selected_node.action_tokens,
                    token_ids=selected_node.action_tokens,
                )
            if observation_text and hasattr(branch.handler, "add_user_message"):
                branch.handler.add_user_message(
                    observation_text,
                    token_ids=observation_tokens,
                )
            if hasattr(branch.handler, "mark_done"):
                branch.handler.mark_done(selected_node.done)

        branch.current_node = selected_node
        branch.done = selected_node.done

    def _build_actor_data(self, roots: Sequence[TreeNode]) -> ActorBatch:
        """把已执行完整轨迹整理为本地 PPO 主样本。"""
        trajectories: list[TrajectoryRecord] = []
        for root in roots:
            for path in self._iter_executed_paths(root):
                trajectory = self._build_trajectory_record_from_path(root=root, path=path)
                trajectories.append(trajectory)
        return ActorBatch(
            trajectories=trajectories,
            metadata={
                "n_trajectories": len(trajectories),
                "n_steps": sum(len(trajectory.steps) for trajectory in trajectories),
                "sum_response_token_weights": sum(
                    sum(trajectory.response_token_weights) for trajectory in trajectories
                ),
            },
        )

    def _build_aux_actor_data(self, aux_samples: Sequence[AuxiliarySample]) -> AuxiliaryBatch:
        """把 sibling 样本整理为本地 auxiliary loss 输入。"""
        return AuxiliaryBatch(
            samples=list(aux_samples),
            metadata={
                "n_samples": len(aux_samples),
                "total_weight": sum(sample.cluster_weight for sample in aux_samples),
                "sum_effective_token_weights": sum(
                    sample.cluster_weight * sample.token_weight for sample in aux_samples
                ),
            },
        )

    def _build_critic_data(self, critic_samples: Sequence[CriticSample]) -> CriticBatch:
        """把 executed transitions 整理为本地 Q-head TD 更新数据。"""
        return CriticBatch(
            samples=list(critic_samples),
            metadata={
                "n_samples": len(critic_samples),
                "mean_reward": (
                    sum(sample.reward for sample in critic_samples) / len(critic_samples)
                    if critic_samples
                    else 0.0
                ),
            },
        )

    def _generate_candidates_for_requests(
        self,
        requests: Sequence[CandidateRequest],
    ) -> dict[Any, list[TreeNode]]:
        if not requests:
            return {}

        prompt_batch: list[list[int]] = []
        prompt_request_indices: list[int] = []
        for request_index, request in enumerate(requests):
            for _ in range(request.budget):
                prompt_batch.append(list(request.state_tokens))
                prompt_request_indices.append(request_index)

        generation_output = self._generate_prompt_batch(prompt_batch)
        outputs_by_prompt = _group_generation_outputs_by_prompt(generation_output, len(prompt_batch))
        outputs_by_request: list[list[Any]] = [[] for _ in requests]

        for prompt_index, request_index in enumerate(prompt_request_indices):
            prompt_outputs = outputs_by_prompt[prompt_index]
            if not prompt_outputs:
                continue
            if len(prompt_outputs) != 1:
                raise ValueError(
                    "inference_engine.generate must return exactly one candidate per prompt "
                    f"when using repeated prompt batching, got {len(prompt_outputs)} outputs "
                    f"for prompt index {prompt_index}"
                )
            outputs_by_request[request_index].append(prompt_outputs[0])

        candidates_by_owner: dict[Any, list[TreeNode]] = {}
        for request, request_outputs in zip(requests, outputs_by_request):
            request.parent.metadata["generation_output"] = list(request_outputs)
            candidates_by_owner[request.owner_id] = self._build_candidate_nodes(
                parent=request.parent,
                state_tokens=request.state_tokens,
                candidate_outputs=request_outputs,
            )
        return candidates_by_owner

    def _build_candidate_nodes(
        self,
        parent: TreeNode,
        state_tokens: Sequence[int],
        candidate_outputs: Sequence[Any],
    ) -> list[TreeNode]:
        candidates: list[TreeNode] = []
        for output in candidate_outputs:
            action_tokens = _resolve_generated_token_ids(output, tokenizer=self.tokenizer)
            if not action_tokens:
                continue
            candidates.append(
                TreeNode(
                    state_tokens=list(state_tokens),
                    action_tokens=action_tokens,
                    parent=parent,
                    depth=parent.depth + 1,
                    state_text=parent.state_text,
                    action_text=_resolve_generated_text(output, tokenizer=self.tokenizer),
                    log_prob=_resolve_log_prob(output),
                    node_id=self._next_node_id("node"),
                    metadata={
                        "generation_output": output,
                    },
                )
            )
        parent.children = candidates
        return candidates

    def _generate_prompt_batch(self, prompt_token_ids: Sequence[Sequence[int]]) -> Any:
        if not prompt_token_ids:
            return []
        return self.inference_engine.generate(
            prompt_token_ids=[list(prompt_ids) for prompt_ids in prompt_token_ids],
        )

    def _collect_auxiliary_samples(self, roots: Sequence[TreeNode]) -> list[AuxiliarySample]:
        auxiliary_samples: list[AuxiliarySample] = []
        for root in roots:
            for parent in _iter_all_nodes([root]):
                if not parent.children:
                    continue
                if parent.parent is None:
                    cluster_labels = parent.metadata.get("cluster_labels")
                    selected_indices = parent.metadata.get("selected_indices")
                    if not cluster_labels or not selected_indices:
                        continue
                    cluster_weights = self.branch_selector.compute_cluster_weights(
                        cluster_labels=cluster_labels,
                        selected_indices=selected_indices,
                    )
                    representative_indices = parent.metadata.get(
                        "representative_indices",
                        selected_indices,
                    )
                    for selected_index in selected_indices:
                        selected_node = parent.children[selected_index]
                        if selected_node.advantage is None:
                            continue
                        selected_cluster_id = cluster_labels[selected_index]
                        auxiliary_indices = [
                            index
                            for index, cluster_id in enumerate(cluster_labels)
                            if cluster_id == selected_cluster_id and index != selected_index
                        ]
                        if not auxiliary_indices:
                            continue
                        selection = SelectionResult(
                            selected_index=selected_index,
                            representative_indices=list(representative_indices),
                            cluster_labels=list(cluster_labels),
                            auxiliary_indices=auxiliary_indices,
                            cluster_weights=cluster_weights,
                        )
                        auxiliary_samples.extend(
                            self.branch_selector.build_auxiliary_samples(
                                parent=parent,
                                candidates=parent.children,
                                selection=selection,
                                advantage=selected_node.advantage,
                                td_target=selected_node.td_target,
                            )
                        )
                    continue

                selection = parent.metadata.get("selection")
                if not isinstance(selection, SelectionResult):
                    continue
                selected_node = parent.children[selection.selected_index]
                if selected_node.advantage is None:
                    continue
                auxiliary_samples.extend(
                    self.branch_selector.build_auxiliary_samples(
                        parent=parent,
                        candidates=parent.children,
                        selection=selection,
                        advantage=selected_node.advantage,
                        td_target=selected_node.td_target,
                    )
                )
        return auxiliary_samples

    def _collect_critic_samples(self, roots: Sequence[TreeNode]) -> list[CriticSample]:
        critic_samples: list[CriticSample] = []
        for node in _iter_all_nodes(roots):
            if not node.executed:
                continue
            critic_samples.append(
                CriticSample(
                    state_tokens=list(node.state_tokens),
                    action_tokens=list(node.action_tokens),
                    reward=float(node.env_reward),
                    next_state_tokens=list(node.next_state_tokens),
                    done=bool(node.done),
                    source_node_id=node.node_id,
                    metadata={
                        "next_state_value": 0.0 if node.done else estimate_state_value(node.children),
                        "td_target": node.td_target,
                    },
                )
            )
        return critic_samples

    def _iter_executed_paths(self, root: TreeNode) -> list[list[TreeNode]]:
        paths: list[list[TreeNode]] = []

        def _dfs(node: TreeNode, path: list[TreeNode]) -> None:
            executed_children = [child for child in node.children if child.executed]
            if not executed_children:
                if path:
                    paths.append(path[:])
                return
            for child in executed_children:
                path.append(child)
                _dfs(child, path)
                path.pop()

        _dfs(root, [])
        return paths

    def _build_trajectory_record_from_path(
        self,
        root: TreeNode,
        path: Sequence[TreeNode],
    ) -> TrajectoryRecord:
        record = TrajectoryRecord(
            input_ids=list(root.state_tokens),
            attention_mask=[1] * len(root.state_tokens),
            position_ids=list(range(len(root.state_tokens))),
            response_mask=[0] * len(root.state_tokens),
            response_token_weights=[0.0] * len(root.state_tokens),
            advantages=[0.0] * len(root.state_tokens),
            returns=[0.0] * len(root.state_tokens),
            state_values=[0.0] * len(root.state_tokens),
            old_log_probs=[0.0] * len(root.state_tokens),
            ref_log_probs=[0.0] * len(root.state_tokens),
            metadata={
                "root_node_id": root.node_id,
                "leaf_node_id": path[-1].node_id if path else root.node_id,
                "flattened": True,
            },
        )

        cursor = len(record.input_ids)
        for node in path:
            action_tokens = list(node.action_tokens)
            observation_tokens = list(node.metadata.get("observation_tokens", []))
            token_weight = resolve_action_token_weight(action_tokens)

            record.steps.append(
                TrajectoryStep(
                    state_tokens=list(node.state_tokens),
                    action_tokens=action_tokens,
                    next_state_tokens=list(node.next_state_tokens),
                    reward=float(node.env_reward),
                    done=bool(node.done),
                    advantage=node.advantage,
                    td_target=node.td_target,
                    state_value=node.state_value,
                    token_weight=token_weight,
                    source_node_id=node.node_id,
                    metadata=dict(node.metadata),
                )
            )

            record.responses.extend(action_tokens)
            record.response_mask.extend([1] * len(action_tokens))
            record.response_token_weights.extend([token_weight] * len(action_tokens))
            record.advantages.extend([float(node.advantage or 0.0)] * len(action_tokens))
            record.returns.extend([float(node.td_target or 0.0)] * len(action_tokens))
            record.state_values.extend([float(node.state_value or 0.0)] * len(action_tokens))
            record.old_log_probs.extend([0.0] * len(action_tokens))
            record.ref_log_probs.extend([0.0] * len(action_tokens))

            record.input_ids.extend(action_tokens)
            record.attention_mask.extend([1] * len(action_tokens))
            record.position_ids.extend(range(cursor, cursor + len(action_tokens)))
            cursor += len(action_tokens)

            if observation_tokens:
                record.input_ids.extend(observation_tokens)
                record.attention_mask.extend([1] * len(observation_tokens))
                record.position_ids.extend(range(cursor, cursor + len(observation_tokens)))
                record.response_mask.extend([0] * len(observation_tokens))
                record.response_token_weights.extend([0.0] * len(observation_tokens))
                record.advantages.extend([0.0] * len(observation_tokens))
                record.returns.extend([0.0] * len(observation_tokens))
                record.state_values.extend([0.0] * len(observation_tokens))
                record.old_log_probs.extend([0.0] * len(observation_tokens))
                record.ref_log_probs.extend([0.0] * len(observation_tokens))
                cursor += len(observation_tokens)

        return record

    def _run_environment_step(self, branch: BranchRuntime, selected_node: TreeNode) -> EnvironmentStep:
        env_client = branch.metadata.get("env_client")
        if env_client is None:
            if (
                selected_node.env_next_state is not None
                or selected_node.next_state_tokens
                or selected_node.done
            ):
                return EnvironmentStep(
                    reward=float(selected_node.env_reward),
                    next_state=selected_node.env_next_state,
                    done=bool(selected_node.done),
                    metadata=dict(selected_node.metadata),
                )
            raise RuntimeError("env_client is required to execute selected action")

        action_payload = selected_node.action_text or self.tokenizer.decode(
            selected_node.action_tokens,
            skip_special_tokens=False,
        )
        raw_step = env_client.step(action_payload)
        return _coerce_environment_step(raw_step)

    def _normalize_prompt_items(self, prompts: Any) -> list[Any]:
        if prompts is None:
            return []
        if isinstance(prompts, SequenceABC) and not isinstance(prompts, (str, bytes, bytearray)):
            return list(prompts)
        return [prompts]

    def _next_node_id(self, prefix: str) -> str:
        self._node_counter += 1
        return f"{prefix}-{self._node_counter}"


def _iter_all_nodes(roots: Sequence[TreeNode]) -> list[TreeNode]:
    nodes: list[TreeNode] = []
    stack = list(reversed(list(roots)))
    while stack:
        node = stack.pop()
        nodes.append(node)
        stack.extend(reversed(node.children))
    return nodes


def _merge_qcritic_outputs_into_nodes(candidates: Sequence[TreeNode], qcritic_output: QCriticOutput) -> None:
    if qcritic_output.q_values is None:
        return
    q_values = qcritic_output.q_values.detach().cpu().tolist()
    for node, q_value in zip(candidates, q_values):
        node.q_value = float(q_value)


def _qcritic_output_to_mapping(qcritic_output: QCriticOutput) -> dict[str, Any]:
    mapping: dict[str, Any] = {
        "action_last_token_indices": list(qcritic_output.action_last_token_indices),
    }
    if qcritic_output.hidden_states is not None:
        mapping["hidden_states"] = qcritic_output.hidden_states
    if qcritic_output.q_values is not None:
        mapping["q_values"] = qcritic_output.q_values
    mapping.update(qcritic_output.metadata)
    return mapping


def _slice_qcritic_output(
    qcritic_output: QCriticOutput,
    start: int,
    end: int,
    total: int,
) -> QCriticOutput:
    metadata = {
        key: _slice_candidate_aligned_value(value, start=start, end=end, total=total)
        for key, value in qcritic_output.metadata.items()
    }
    hidden_states = _slice_candidate_aligned_value(qcritic_output.hidden_states, start=start, end=end, total=total)
    q_values = _slice_candidate_aligned_value(qcritic_output.q_values, start=start, end=end, total=total)
    action_last_token_indices = list(qcritic_output.action_last_token_indices[start:end])
    return QCriticOutput(
        hidden_states=hidden_states,
        q_values=q_values,
        action_last_token_indices=action_last_token_indices,
        metadata=metadata,
    )


def _slice_candidate_aligned_value(value: Any, start: int, end: int, total: int) -> Any:
    if value is None:
        return None
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) > 0 and int(shape[0]) == total:
        return value[start:end]
    if isinstance(value, (list, tuple)) and len(value) == total:
        return value[start:end]
    return value


def _group_generation_outputs_by_prompt(generation_output: Any, n_prompts: int) -> list[list[Any]]:
    if n_prompts == 0:
        return []

    nested_outputs = getattr(generation_output, "outputs", None)
    if nested_outputs is None and isinstance(generation_output, Mapping):
        nested_outputs = generation_output.get("outputs")
    if isinstance(nested_outputs, SequenceABC) and not isinstance(nested_outputs, (str, bytes, bytearray)):
        sequence = list(nested_outputs)
        if len(sequence) == n_prompts:
            return [_flatten_generation_outputs(item) or [item] for item in sequence]

    if isinstance(generation_output, SequenceABC) and not isinstance(
        generation_output,
        (str, bytes, bytearray),
    ):
        sequence = list(generation_output)
        if len(sequence) == n_prompts:
            return [
                _flatten_generation_outputs(item) or [item]
                for item in sequence
            ]

    if n_prompts == 1:
        flattened = _flatten_generation_outputs(generation_output)
        return [flattened or [generation_output]]

    flattened = _flatten_generation_outputs(generation_output)
    if len(flattened) == n_prompts:
        return [[item] for item in flattened]

    raise ValueError(
        "failed to align generation outputs with prompt batch: "
        f"expected {n_prompts} prompt-level outputs, got {len(flattened)} flattened outputs"
    )


def _flatten_generation_outputs(generation_output: Any) -> list[Any]:
    if generation_output is None:
        return []
    if isinstance(generation_output, Mapping):
        if "outputs" in generation_output:
            return _flatten_generation_outputs(generation_output["outputs"])
        return [generation_output]
    if hasattr(generation_output, "outputs"):
        outputs = getattr(generation_output, "outputs")
        flattened = _flatten_generation_outputs(outputs)
        return flattened if flattened else [generation_output]
    if isinstance(generation_output, SequenceABC) and not isinstance(generation_output, (str, bytes, bytearray)):
        if generation_output and all(isinstance(item, int) for item in generation_output):
            return [list(generation_output)]
        flattened: list[Any] = []
        for item in generation_output:
            nested = _flatten_generation_outputs(item)
            flattened.extend(nested if nested else [item])
        return flattened
    return [generation_output]


def _resolve_token_ids(value: Any, tokenizer: TokenizerProtocol) -> list[int]:
    for key in ("state_tokens", "input_ids", "prompt_token_ids", "token_ids"):
        candidate = _resolve_field(value, key)
        if candidate is None:
            continue
        if isinstance(candidate, SequenceABC) and not isinstance(candidate, (str, bytes, bytearray)):
            return [int(token_id) for token_id in candidate]

    text = _resolve_text(value)
    return _encode_text(tokenizer, text)


def _resolve_generated_token_ids(value: Any, tokenizer: TokenizerProtocol) -> list[int]:
    for key in ("action_tokens", "token_ids", "output_token_ids", "response_token_ids"):
        candidate = _resolve_field(value, key)
        if candidate is None:
            continue
        if isinstance(candidate, SequenceABC) and not isinstance(candidate, (str, bytes, bytearray)):
            return [int(token_id) for token_id in candidate]

    text = _resolve_generated_text(value, tokenizer=tokenizer)
    return _encode_text(tokenizer, text)


def _resolve_generated_text(value: Any, tokenizer: TokenizerProtocol) -> str:
    for key in ("action_text", "text", "output_text", "response_text"):
        candidate = _resolve_field(value, key)
        if isinstance(candidate, str):
            return candidate
    token_ids = _resolve_field(value, "token_ids")
    if isinstance(token_ids, SequenceABC) and not isinstance(token_ids, (str, bytes, bytearray)):
        return tokenizer.decode(token_ids, skip_special_tokens=False)
    return ""


def _resolve_text(value: Any) -> str:
    for key in ("state_text", "prompt", "text", "observation", "content"):
        candidate = _resolve_field(value, key)
        if isinstance(candidate, str):
            return candidate
    return ""


def _resolve_log_prob(value: Any) -> float | None:
    for key in ("log_prob", "cum_logprob", "cumulative_logprob", "score"):
        candidate = _resolve_field(value, key)
        if candidate is None:
            continue
        try:
            return float(candidate)
        except (TypeError, ValueError):
            continue
    return None


def _resolve_field(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _resolve_mapping(value: Any, key: str) -> dict[str, Any] | None:
    candidate = _resolve_field(value, key)
    if isinstance(candidate, Mapping):
        return dict(candidate)
    return None


def _encode_text(tokenizer: TokenizerProtocol, text: str) -> list[int]:
    if not text:
        return []
    return list(tokenizer.encode(text, add_special_tokens=False))


def _coerce_environment_step(value: Any) -> EnvironmentStep:
    if isinstance(value, tuple) and len(value) == 4:
        next_state, reward, done, metadata = value
        return EnvironmentStep(
            reward=float(reward),
            next_state=next_state,
            done=bool(done),
            metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
        )
    raise TypeError("environment step result must be a 4-tuple: (next_state, reward, done, metadata)")


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in ("text", "observation", "content", "message"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                return candidate
    return str(value)
