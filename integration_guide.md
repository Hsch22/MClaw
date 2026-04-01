# MClaw Integration Guide

## 0. Conventions

- Training env: `mclaw-train`
- Env server env: `mclaw-envserver`
- Repo root: `/mnt/kangshijia/husicheng/MClaw`

All commands below assume:

```bash
cd /mnt/kangshijia/husicheng/MClaw
```

**Override syntax note**: Parameters under the `mclaw:` namespace in the YAML
(i.e. `tree_rollout`, `clustering`, `q_critic`, `aux_loss`) must use the
`mclaw.` prefix when passed as CLI overrides. Top-level keys (`distributed`,
`trainer`, `data`, `adapter`, `model`, `logging`, `actor_rollout_ref`,
`environment`, `algorithm`) do not need a prefix.

## 1. Basic Environment Check

Confirm Python / torch / CUDA:

```bash
/mnt/kangshijia/wangbinyu/conda_envs/mclaw-train/bin/python - <<'PY'
import sys, torch
print("python:", sys.executable)
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
PY
```

Confirm key dependencies:

```bash
PYTHONPATH=/mnt/kangshijia/husicheng/MClaw \
/mnt/kangshijia/wangbinyu/conda_envs/mclaw-train/bin/python - <<'PY'
mods = [
    "torch",
    "transformers",
    "vllm",
    "omegaconf",
    "wandb",
    "tensorboard",
]
for name in mods:
    try:
        __import__(name)
        print(name, "OK")
    except Exception as e:
        print(name, "ERR", repr(e))
PY
```

If connecting to real verl, also check:

```bash
PYTHONPATH=/mnt/kangshijia/husicheng/MClaw:/mnt/kangshijia/husicheng/verl \
/mnt/kangshijia/wangbinyu/conda_envs/mclaw-train/bin/python - <<'PY'
mods = [
    "verl",
    "ray",
]
for name in mods:
    try:
        __import__(name)
        print(name, "OK")
    except Exception as e:
        print(name, "ERR", repr(e))
PY
```

## 2. Compile Check

```bash
PYTHONPATH=/mnt/kangshijia/husicheng/MClaw \
/mnt/kangshijia/wangbinyu/conda_envs/mclaw-train/bin/python -m compileall mclaw tests
```

## 3. Regression Tests

```bash
PYTHONPATH=/mnt/kangshijia/husicheng/MClaw \
/mnt/kangshijia/wangbinyu/conda_envs/mclaw-train/bin/python -m unittest tests.test_regressions
```

## 4. Config Check

Expand the final config and confirm key fields:

```bash
PYTHONPATH=/mnt/kangshijia/husicheng/MClaw \
/mnt/kangshijia/wangbinyu/conda_envs/mclaw-train/bin/python - <<'PY'
from mclaw.trainer.main import load_config
cfg = load_config("mclaw/config/mclaw_trainer.yaml", [])
print("model_path =", cfg.model.model_path)
print("model_family =", cfg.model.family)
print("tokenizer_path =", cfg.model.tokenizer_path)
print("actor_backend =", cfg.adapter.actor_backend)
print("inference_engine =", cfg.adapter.inference_engine)
print("rollout_handler =", cfg.adapter.rollout_handler)
print("env_client =", cfg.adapter.env_client)
print("tracker =", cfg.logging.tracker)
print("enable_fsdp =", cfg.distributed.enable_fsdp)
print("ppo_epochs =", cfg.actor_rollout_ref.get("actor", {}).get("ppo_epochs"))
PY
```

To override config via CLI, note the `mclaw.` prefix for nested keys:

```bash
PYTHONPATH=/mnt/kangshijia/husicheng/MClaw \
/mnt/kangshijia/wangbinyu/conda_envs/mclaw-train/bin/python -m mclaw.trainer.main \
  --config mclaw/config/mclaw_trainer.yaml \
  model.model_path=YOUR_MODEL \
  model.tokenizer_path=YOUR_TOKENIZER \
  data.train_file=YOUR_DATA \
  adapter.task_name=YOUR_TASK \
  adapter.env_addr=YOUR_ENV_ADDR
```

## 5. Build Trainer Only (No Training)

Confirm the build phase passes. **Prerequisite**: `model.family` or
`model.model_path` must point to a valid, accessible model directory.

```bash
PYTHONPATH=/mnt/kangshijia/husicheng/MClaw:/mnt/kangshijia/husicheng/verl \
/mnt/kangshijia/wangbinyu/conda_envs/mclaw-train/bin/python - <<'PY'
from mclaw.trainer.main import load_config, build_trainer
cfg = load_config("mclaw/config/mclaw_trainer.yaml", [])
trainer = build_trainer(cfg)
print("trainer OK")
print("tree_rollout =", type(trainer.tree_rollout).__name__)
print("actor =", type(trainer.actor).__name__ if trainer.actor else None)
print("ref_policy =", type(trainer.ref_policy).__name__ if trainer.ref_policy else None)
print("logger =", type(trainer.logger).__name__ if trainer.logger else None)
PY
```

If this fails, fix import / config path / GPU / verl+ray dependency issues first.

## 6. Single-Prompt Smoke Test

Goal: run one prompt to verify:

- `TreeRollout` generates candidates
- `handler` multi-turn prompt participates
- `auxiliary_batch` is non-empty and doesn't crash
- `update_policy()` with joint PPO + auxiliary completes one step

```bash
PYTHONPATH=/mnt/kangshijia/husicheng/MClaw:/mnt/kangshijia/husicheng/verl \
/mnt/kangshijia/wangbinyu/conda_envs/mclaw-train/bin/python - <<'PY'
from mclaw.trainer.main import load_config, build_trainer

cfg = load_config("mclaw/config/mclaw_trainer.yaml", [])
# Shrink rollout budget for fast smoke test
cfg.trainer.max_steps = 1
cfg.data.train_batch_size = 1
cfg.tree_rollout.root_budget = 4
cfg.tree_rollout.n_envs = 2
cfg.tree_rollout.root_clusters = 2
cfg.tree_rollout.branch_budget = 2
cfg.tree_rollout.intra_branch_clusters = 1
cfg.tree_rollout.max_rounds = 2

trainer = build_trainer(cfg)
print("build done")
trainer.fit()
print("fit done")
PY
```

If the real environment is too heavy, point `data.train_file` to a file with
only 1 sample.

## 7. Environment Server Integration

If AgentEnv requires a separate server, start the server first.

Server env check:

```bash
/mnt/kangshijia/wangbinyu/conda_envs/mclaw-envserver/bin/python - <<'PY'
import sys, torch
print(sys.executable)
print(torch.__version__)
PY
```

When starting the env server, note down:

- Listening address
- task_name
- Required reset parameters

Then align on the training side:

```bash
adapter.task_name=...
adapter.env_addr=...
environment.reset_kwargs.xxx=...
```

Redirect server and training logs separately:

```bash
mkdir -p /mnt/kangshijia/husicheng/logs
```

## 8. verl + vLLM Single-GPU (No FSDP)

```bash
PYTHONPATH=/mnt/kangshijia/husicheng/MClaw:/mnt/kangshijia/husicheng/verl \
CUDA_VISIBLE_DEVICES=0 \
/mnt/kangshijia/wangbinyu/conda_envs/mclaw-train/bin/python -m mclaw.trainer.main \
  --config mclaw/config/mclaw_trainer.yaml \
  distributed.enable_fsdp=false \
  trainer.max_steps=1 \
  data.train_batch_size=1
```

Focus areas:

- `VerlInferenceEngine` launches vLLM successfully
- `VerlActorBackend` calls `compute_log_prob` / `update_policy` without error
- Joint PPO + auxiliary path has no shape/device mismatches

## 9. FSDP Multi-GPU

```bash
cd /mnt/kangshijia/husicheng/MClaw
PYTHONPATH=/mnt/kangshijia/husicheng/MClaw:/mnt/kangshijia/husicheng/verl \
CUDA_VISIBLE_DEVICES=0,1 \
/mnt/kangshijia/wangbinyu/conda_envs/mclaw-train/bin/torchrun \
  --nproc_per_node=2 \
  -m mclaw.trainer.main \
  --config mclaw/config/mclaw_trainer.yaml \
  distributed.enable_fsdp=true \
  distributed.tensor_parallel_size=1 \
  trainer.max_steps=1 \
  data.train_batch_size=1
```

Focus areas:

- Rollout sharding manager enters/exits cleanly
- QCritic reuses FSDP backbone forward correctly
- Checkpoint save/load works

## 10. Checkpoint Integration

Generate a checkpoint:

```bash
PYTHONPATH=/mnt/kangshijia/husicheng/MClaw:/mnt/kangshijia/husicheng/verl \
CUDA_VISIBLE_DEVICES=0 \
/mnt/kangshijia/wangbinyu/conda_envs/mclaw-train/bin/python -m mclaw.trainer.main \
  --config mclaw/config/mclaw_trainer.yaml \
  distributed.enable_fsdp=false \
  trainer.max_steps=2 \
  trainer.save_freq=1 \
  trainer.checkpoint_dir=/mnt/kangshijia/husicheng/MClaw/tmp_ckpt
```

Verify output:

```bash
ls -lah /mnt/kangshijia/husicheng/MClaw/tmp_ckpt
```

Resume from latest checkpoint:

```bash
PYTHONPATH=/mnt/kangshijia/husicheng/MClaw:/mnt/kangshijia/husicheng/verl \
CUDA_VISIBLE_DEVICES=0 \
/mnt/kangshijia/wangbinyu/conda_envs/mclaw-train/bin/python -m mclaw.trainer.main \
  --config mclaw/config/mclaw_trainer.yaml \
  --resume /mnt/kangshijia/husicheng/MClaw/tmp_ckpt
```

Repeat under FSDP as well.

## 11. Tracker Integration

TensorBoard:

```bash
PYTHONPATH=/mnt/kangshijia/husicheng/MClaw:/mnt/kangshijia/husicheng/verl \
CUDA_VISIBLE_DEVICES=0 \
/mnt/kangshijia/wangbinyu/conda_envs/mclaw-train/bin/python -m mclaw.trainer.main \
  --config mclaw/config/mclaw_trainer.yaml \
  logging.tracker=tensorboard \
  logging.experiment_name=debug_tb \
  trainer.max_steps=1
```

Verify files:

```bash
find /mnt/kangshijia/husicheng/MClaw/checkpoints/tensorboard -maxdepth 3 -type f | head
```

WandB:

```bash
PYTHONPATH=/mnt/kangshijia/husicheng/MClaw:/mnt/kangshijia/husicheng/verl \
CUDA_VISIBLE_DEVICES=0 \
/mnt/kangshijia/wangbinyu/conda_envs/mclaw-train/bin/python -m mclaw.trainer.main \
  --config mclaw/config/mclaw_trainer.yaml \
  logging.tracker=wandb \
  logging.project_name=mclaw \
  logging.experiment_name=debug_wandb \
  trainer.max_steps=1
```

## 12. Key Metrics to Watch

Rollout health:
- `rollout/n_roots`
- `rollout/n_aux_samples`
- `rollout/n_actor_trajectories`

Actor training:
- `actor/pg_loss`
- `actor/aux_loss`
- `actor/aux_mean_log_prob` — avg log-prob of auxiliary samples; check aux loss is effective
- `actor/aux_effective_tokens` — token count in aux loss
- `actor/aux_weight_abs_sum` — if 0, all advantages are 0 and aux loss is no-op
- `actor/grad_norm`

Critic:
- `critic/q_loss`

If `rollout/n_aux_samples > 0` but `actor/aux_loss` is always 0, check whether
the joint update path (`_update_policy_with_auxiliary`) is being reached.

## 13. Common Failure Points (In Debugging Order)

1. `ray` / `verl` / `vllm` import failure
2. Model path or tokenizer path invalid / inaccessible
3. `adapter.task_name` / `adapter.env_addr` not configured
4. vLLM startup failure or CUDA not visible
5. DataProto tensor shape mismatch
6. Joint PPO + auxiliary path device / dtype mismatch
7. FSDP state_dict / optimizer_state_dict restore failure

## 14. Recommended Integration Order

1. Compile check (`compileall`)
2. Regression tests (`unittest`)
3. Build trainer only
4. Single-GPU, no FSDP, 1-step smoke test
5. Connect real env server
6. Single-GPU real rollout + update
7. Multi-GPU FSDP
8. Checkpoint save/load
9. Tracker

## 15. Minimal End-to-End Integration Command

```bash
cd /mnt/kangshijia/husicheng/MClaw
PYTHONPATH=/mnt/kangshijia/husicheng/MClaw:/mnt/kangshijia/husicheng/verl \
CUDA_VISIBLE_DEVICES=0 \
/mnt/kangshijia/wangbinyu/conda_envs/mclaw-train/bin/python -m mclaw.trainer.main \
  --config mclaw/config/mclaw_trainer.yaml \
  distributed.enable_fsdp=false \
  trainer.max_steps=1 \
  data.train_batch_size=1 \
  mclaw.tree_rollout.root_budget=4 \
  mclaw.tree_rollout.n_envs=2 \
  mclaw.tree_rollout.root_clusters=2 \
  mclaw.tree_rollout.branch_budget=2 \
  mclaw.tree_rollout.intra_branch_clusters=1 \
  mclaw.tree_rollout.max_rounds=2
```
