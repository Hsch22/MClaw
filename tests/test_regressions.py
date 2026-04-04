from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from mclaw.adapters import DataProtoAdapter, StandardLogger, VerlActorBackend, build_tracker
from mclaw.clustering import ActionClusterer
from mclaw.clustering.base import BaseClusterer, ClusterResult
from mclaw.config import ClusteringConfig, MClawTrainerConfig, TreeRolloutConfig
from mclaw.core.contracts import ActorBatch, AuxiliaryBatch, AuxiliarySample, EnvironmentStep, TrajectoryRecord
from mclaw.core.tree_node import TreeNode
from mclaw.core.tree_rollout import TreeRollout
from mclaw.critic import compute_tree_advantage
from mclaw.critic.q_critic import QCriticOutput
from mclaw.trainer.main import _build_clusterer
from mclaw.trainer.mclaw_trainer import MClawTrainer, _validate_load_state_result


class _FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False, **kwargs: object) -> list[int]:
        del add_special_tokens, kwargs
        return [len(text)]

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = False,
        **kwargs: object,
    ) -> str:
        del skip_special_tokens, kwargs
        return " ".join(str(token_id) for token_id in token_ids)


class _MockHandler:
    def __init__(self, root_prompt: list[int], next_prompt: list[int]) -> None:
        self.root_prompt = list(root_prompt)
        self.next_prompt = list(next_prompt)
        self.stage = 0
        self.score = 0.0
        self.done = False

    def clone(self) -> "_MockHandler":
        cloned = _MockHandler(self.root_prompt, self.next_prompt)
        cloned.stage = self.stage
        cloned.score = self.score
        cloned.done = self.done
        return cloned

    def get_generation_prompt(self, tokenizer: object) -> list[int]:
        del tokenizer
        return self.root_prompt if self.stage == 0 else self.next_prompt

    def add_assistant_message(self, action: object, token_ids: list[int] | None = None) -> None:
        del action, token_ids
        self.stage = 1

    def add_user_message(self, observation: object, token_ids: list[int] | None = None) -> None:
        del observation, token_ids
        self.stage = 1

    def mark_done(self, done: bool) -> None:
        self.done = bool(done)


class _FakeInferenceEngine:
    def __init__(self) -> None:
        self.calls: list[list[list[int]]] = []
        self._token = 10

    def generate(self, prompt_token_ids: list[list[int]], **kwargs: object) -> list[dict[str, list[int]]]:
        del kwargs
        self.calls.append([list(row) for row in prompt_token_ids])
        outputs = []
        for _ in prompt_token_ids:
            self._token += 1
            outputs.append({"token_ids": [self._token], "text": str(self._token), "log_prob": 0.0})
        return outputs


class _FakeClusterer(BaseClusterer):
    def __init__(self) -> None:
        super().__init__(ClusteringConfig())

    def extract_features(self, action_token_ids, state_token_ids, model_outputs):  # type: ignore[override]
        del action_token_ids, state_token_ids, model_outputs
        return torch.ones(1, 1)

    def cluster_candidates(self, nodes, n_clusters, model_outputs):  # type: ignore[override]
        del n_clusters, model_outputs
        return ClusterResult(
            labels=[0 for _ in nodes],
            representative_indices=[0],
        )


class _FakeQCritic:
    def __init__(self) -> None:
        self.config = SimpleNamespace(gamma=0.99)

    def score_actions(self, state_token_ids, action_token_ids):  # type: ignore[override]
        del state_token_ids
        return QCriticOutput(q_values=torch.ones(len(action_token_ids), dtype=torch.float32))


class _FakeEnvClient:
    def __init__(self) -> None:
        self.steps = 0

    def reset(self, item_id: object, **kwargs: object) -> None:
        del item_id, kwargs

    def step(self, action: object, **kwargs: object) -> tuple[str, float, bool, dict[str, object]]:
        del action, kwargs
        self.steps += 1
        return (
            f"obs-{self.steps}",
            1.0,
            self.steps >= 2,
            {},
        )


class _ToyLM(nn.Module):
    def __init__(self, vocab_size: int = 32, hidden_dim: int = 8) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, attention_mask=None, return_dict: bool = True):  # type: ignore[override]
        del attention_mask, return_dict
        hidden = self.embedding(input_ids)
        logits = self.proj(hidden)
        return SimpleNamespace(logits=logits)


@dataclass
class _LoadResult:
    missing_keys: list[str]
    unexpected_keys: list[str]


class _CountingActorBackend:
    def __init__(self) -> None:
        self.compute_log_prob_calls = 0
        self.update_policy_calls = 0
        self.update_aux_loss_calls = 0

    def compute_log_prob(self, batch: object) -> dict[str, list[list[float]]]:
        del batch
        self.compute_log_prob_calls += 1
        return {"old_log_probs": [[0.0, 0.0]]}

    def update_policy(self, batch: object) -> dict[str, float]:
        del batch
        self.update_policy_calls += 1
        return {"actor/pg_loss": 1.0}

    def update_aux_loss(self, batch: object) -> dict[str, float]:
        del batch
        self.update_aux_loss_calls += 1
        return {"actor/aux_loss": 1.0}


class RegressionTests(unittest.TestCase):
    def test_tree_rollout_uses_handler_generation_prompt(self) -> None:
        engine = _FakeInferenceEngine()
        rollout = TreeRollout(
            inference_engine=engine,
            actor_module_fsdp=None,
            q_critic=_FakeQCritic(),
            clusterer=_FakeClusterer(),
            tokenizer=_FakeTokenizer(),
            config=TreeRolloutConfig(
                root_budget=1,
                n_envs=1,
                root_clusters=1,
                branch_budget=1,
                intra_branch_clusters=1,
                max_rounds=2,
            ),
            env_client_factory=_FakeEnvClient,
            handler_factory=lambda prompt_item, root: _MockHandler([101], [202]),
        )

        rollout.generate_tree_rollout([{"prompt_token_ids": [1, 2, 3], "item_id": "x"}])

        self.assertEqual(engine.calls[0], [[101]])
        self.assertEqual(engine.calls[1], [[202]])

    def test_compute_tree_advantage_uses_stable_parent_baseline(self) -> None:
        root = TreeNode(state_tokens=[0], node_id="root")
        first = TreeNode(
            state_tokens=[0],
            parent=root,
            depth=1,
            executed=True,
            env_reward=10.0,
            done=True,
            q_value=1.0,
            node_id="node-1",
        )
        second = TreeNode(
            state_tokens=[0],
            parent=root,
            depth=1,
            executed=True,
            env_reward=20.0,
            done=True,
            q_value=3.0,
            node_id="node-2",
        )
        root.children = [first, second]

        result = compute_tree_advantage([root], gamma=0.9)

        self.assertEqual(result.metadata["n_executed_nodes"], 2)
        self.assertAlmostEqual(root.state_value or 0.0, 2.0)
        self.assertAlmostEqual(first.state_value or 0.0, 2.0)
        self.assertAlmostEqual(second.state_value or 0.0, 2.0)
        self.assertAlmostEqual(first.td_target or 0.0, 10.0)
        self.assertAlmostEqual(second.td_target or 0.0, 20.0)
        self.assertAlmostEqual(first.advantage or 0.0, 8.0)
        self.assertAlmostEqual(second.advantage or 0.0, 18.0)

    def test_tree_rollout_record_responses_exclude_observations(self) -> None:
        engine = _FakeInferenceEngine()
        rollout = TreeRollout(
            inference_engine=engine,
            actor_module_fsdp=None,
            q_critic=_FakeQCritic(),
            clusterer=_FakeClusterer(),
            tokenizer=_FakeTokenizer(),
            config=TreeRolloutConfig(
                root_budget=1,
                n_envs=1,
                root_clusters=1,
                branch_budget=1,
                intra_branch_clusters=1,
                max_rounds=2,
            ),
            env_client_factory=_FakeEnvClient,
        )

        output = rollout.generate_tree_rollout([{"prompt_token_ids": [1, 2, 3], "item_id": "x"}])

        trajectory = output.actor_data.trajectories[0]
        prompt_length = trajectory.metadata["prompt_length"]
        expected_responses = [
            token_id
            for token_id, response_mask in zip(trajectory.input_ids, trajectory.response_mask)
            if response_mask
        ]
        self.assertEqual(trajectory.responses, expected_responses)
        self.assertNotEqual(trajectory.responses, trajectory.input_ids[prompt_length:])
        self.assertIn(0, trajectory.response_mask[prompt_length:])

    def test_clusterer_reseeds_empty_clusters_when_donor_pool_is_unavailable(self) -> None:
        clusterer = _FakeClusterer()
        labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        distances = torch.tensor(
            [
                [0.1, 0.8, 1.4, 1.5],
                [0.2, 0.7, 1.6, 1.7],
                [0.9, 0.1, 1.8, 1.9],
                [1.0, 0.2, 2.0, 2.1],
            ],
            dtype=torch.float32,
        )

        with patch.object(clusterer, "_collect_donor_indices", return_value=[]):
            adjusted = clusterer._ensure_non_empty_clusters(labels, distances, 4)

        counts = torch.bincount(adjusted, minlength=4)
        self.assertTrue(torch.all(counts > 0).item())

    def test_action_clusterer_uses_raw_action_tokens_without_model_outputs(self) -> None:
        clusterer = ActionClusterer(ClusteringConfig(pca_dim=0))

        features = clusterer.extract_features(
            action_token_ids=[[3, 0], [3], [8, 9]],
            state_token_ids=[[1], [1], [1]],
            model_outputs={},
        )

        expected = torch.tensor(
            [
                [4.0, 1.0],
                [4.0, 0.0],
                [9.0, 10.0],
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.equal(features, expected))

    def test_action_clusterer_groups_identical_token_sequences_exactly(self) -> None:
        clusterer = ActionClusterer(ClusteringConfig(method="action", pca_dim=128))
        nodes = [
            TreeNode(state_tokens=[1], action_tokens=[7, 8]),
            TreeNode(state_tokens=[1], action_tokens=[7, 8]),
            TreeNode(state_tokens=[1], action_tokens=[9]),
            TreeNode(state_tokens=[1], action_tokens=[7, 8]),
            TreeNode(state_tokens=[1], action_tokens=[9]),
        ]

        result = clusterer.cluster_candidates(nodes, n_clusters=1, model_outputs={})

        self.assertEqual(result.labels, [0, 0, 1, 0, 1])
        self.assertEqual(result.representative_indices, [0, 2])
        self.assertEqual(result.metadata["n_clusters"], 2)
        self.assertTrue(result.metadata["requested_clusters_ignored"])
        self.assertEqual(result.metadata["cluster_algorithm"], "exact_action_token_match")

    def test_build_clusterer_supports_action_method(self) -> None:
        clusterer = _build_clusterer(
            MClawTrainerConfig(clustering=ClusteringConfig(method="action"))
        )

        self.assertIsInstance(clusterer, ActionClusterer)

    def test_auxiliary_loss_updates_actor_parameters(self) -> None:
        torch.manual_seed(0)
        actor_module = _ToyLM()
        optimizer = torch.optim.SGD(actor_module.parameters(), lr=0.1)
        backend = VerlActorBackend(
            actor=SimpleNamespace(
                actor_module=actor_module,
                actor_optimizer=optimizer,
                config=SimpleNamespace(use_kl_loss=False),
            ),
            adapter=DataProtoAdapter(pad_token_id=0),
            dataproto_meta_info={"micro_batch_size": 1},
            aux_loss_config={"coef": 1.0, "use_same_advantage": True},
        )
        before = [parameter.detach().clone() for parameter in actor_module.parameters()]

        metrics = backend.update_aux_loss(
            AuxiliaryBatch(
                samples=[
                    AuxiliarySample(
                        state_tokens=[1, 2],
                        action_tokens=[3, 4],
                        advantage=1.0,
                        cluster_weight=1.0,
                        token_weight=0.5,
                    )
                ]
            )
        )

        self.assertIn("actor/aux_loss", metrics)
        self.assertGreater(metrics["actor/aux_samples"], 0.0)
        self.assertTrue(
            any(
                not torch.allclose(before_param, after_param.detach())
                for before_param, after_param in zip(before, actor_module.parameters())
            )
        )

    def test_build_tracker_supports_tensorboard_backend(self) -> None:
        class _FakeWriter:
            def __init__(self, log_dir: str) -> None:
                self.log_dir = log_dir
                self.scalars: list[tuple[str, float, int | None]] = []

            def add_scalar(self, key: str, value: float, global_step: int | None = None) -> None:
                self.scalars.append((key, value, global_step))

            def flush(self) -> None:
                return None

            def close(self) -> None:
                return None

        with patch("mclaw.adapters.logger._import_summary_writer", return_value=_FakeWriter):
            tracker = build_tracker(
                tracker_name="tensorboard",
                project_name="proj",
                experiment_name="exp",
                default_local_dir="/tmp/mclaw-tests",
                path_pattern="log/{timestamp}.log",
                tracker_kwargs=None,
            )

        logger = StandardLogger(tracker=tracker, python_logger=None)
        logger.log({"metric": 1.0}, step=3)
        writer = tracker.writer
        self.assertEqual(writer.scalars, [("metric", 1.0, 3)])

    def test_validate_load_state_result_raises_on_key_mismatch(self) -> None:
        with self.assertRaises(RuntimeError):
            _validate_load_state_result(
                _LoadResult(missing_keys=["weight"], unexpected_keys=[]),
                module=nn.Linear(2, 2),
            )

    def test_trainer_update_actor_calls_backend_once_and_skips_standalone_aux_update(self) -> None:
        backend = _CountingActorBackend()
        trainer_config = MClawTrainerConfig(
            actor_rollout_ref={"actor": {"ppo_epochs": 3}},
        )
        trainer = MClawTrainer(
            config=trainer_config,
            actor=backend,
        )
        actor_data = ActorBatch(
            trajectories=[
                TrajectoryRecord(
                    input_ids=[1, 2],
                    attention_mask=[1, 1],
                    position_ids=[0, 1],
                    response_mask=[0, 1],
                    response_token_weights=[0.0, 1.0],
                    advantages=[0.0, 1.0],
                    returns=[0.0, 1.0],
                    state_values=[0.0, 0.0],
                    old_log_probs=[0.0, 0.0],
                    ref_log_probs=[0.0, 0.0],
                    metadata={"prompt_length": 1},
                )
            ]
        )
        aux_data = AuxiliaryBatch(
            samples=[
                AuxiliarySample(
                    state_tokens=[1],
                    action_tokens=[2],
                    advantage=1.0,
                    cluster_weight=1.0,
                )
            ]
        )

        metrics = trainer.update_actor(actor_data, aux_actor_data=aux_data)

        self.assertEqual(backend.compute_log_prob_calls, 1)
        self.assertEqual(backend.update_policy_calls, 1)
        self.assertEqual(backend.update_aux_loss_calls, 0)
        self.assertEqual(metrics["actor/ppo_epochs"], 3.0)


if __name__ == "__main__":
    unittest.main()
