# MClaw

## 环境（Conda）

- `agentgym-rl-qwen3-clean`：`/mnt/kangshijia/wangbinyu/conda_envs/agentgym-rl-qwen3-clean`
- `agentenv-webarena`：`/mnt/kangshijia/wangbinyu/conda_envs/agentenv-webarena`
- 与 AgentGym-RL / `verl` 联调时常用：`mclaw-train` → `/mnt/kangshijia/wangbinyu/conda_envs/mclaw-train`（需在该环境中安装 `verl`、`vllm` 等栈）

## CUDA Toolkit、`CUDA_HOME` 与 `nvcc`

项目根目录（`MClaw` 的上级）下常放置与 PyTorch **`cu124`** 对齐的 **CUDA 12.4** 工具链，例如：

`/mnt/kangshijia/husicheng/.local/cuda-12.4`

该路径下的 `nvcc` **默认不在系统 `PATH`**，新开终端需先导出再编译 CUDA 扩展或安装会从源码构建的包：

```bash
export CUDA_HOME=/mnt/kangshijia/husicheng/.local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
nvcc --version   # 应显示 release 12.4
```

不建议用 `sudo apt install nvidia-cuda-toolkit` 与上述 12.4 混用，除非整台机统一改用发行版提供的 Toolkit 版本。

若未设置 `CUDA_HOME` 就执行 `pip install flash-attn` 等，可能在元数据阶段失败，提示 `CUDA_HOME environment variable is not set` 或找不到 `nvcc`。

## flash-attn（`verl` / `DataParallelPPOActor` 等）

部分依赖会 `import flash_attn`。任选其一即可。

**1. 预编译 wheel（推荐，可不配置 `nvcc`）**

先确认版本：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

示例（Torch 2.6 + CUDA 12.x + Python 3.10；版本与文件名以 [flash-attention Releases](https://github.com/Dao-AILab/flash-attention/releases) 为准）：

```bash
pip install flash-attn --no-build-isolation -f https://github.com/Dao-AILab/flash-attention/releases/expanded_assets/v2.7.4.post1
```

若仍触发源码构建，从 Release 页选择匹配 `cu12`、`torch2.6`、`cp310` 以及 **cxx11 ABI** 的 `.whl` 直链安装。ABI 检查：

```bash
python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"
```

**2. 源码编译**

在同一 shell 中先完成上文「`CUDA_HOME` + `PATH` + `LD_LIBRARY_PATH`」，且 `nvcc --version` 正常后：

```bash
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

## vLLM 注意力后端（可选）

与仓库根目录 `single-machine.md`、`script/b_train_textcraft.sh` 等一致时，可用 XFormers 避免走 FlashAttention 内核：

```bash
export VLLM_USE_MODELSCOPE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=XFORMERS
```

## 自检

```bash
python -c "from flash_attn.bert_padding import pad_input; print('flash-attn OK')"
python -c "from verl.workers.agent_actor.dp_actor import DataParallelPPOActor; print('verl dp_actor OK')"
```

---

MClaw 是一个面向 Agent RL 的树状 rollout 框架。核心逻辑和训练链路已可装配运行。

## 当前实现阶段

- 已实现：
  - 树状 rollout 主流程：根节点批量展开、后续分支展开、候选动作批量打分、聚类、真实执行、轨迹展平。
  - 向量类聚类基类：可选 PCA 降维 + 确定性的 torch K-Means + representative 选择。
  - 五类聚类方法：`action`（精确 token 分组）、`hidden_state`、`output_grad`、`logprob`、`logit_distribution`。
  - Q-head / Q-critic：共享 backbone 编码 `[state + action]`，只更新 Q-head。
  - 树上的 step-level advantage：V(s) 预缓存避免兄弟节点间 q_value 覆写污染。
  - `load_config()`、`build_trainer()`、`main()` 入口已完成，支持 CLI override。
  - FSDP-aware checkpoint：`FullStateDictConfig` + rank-0 only saving。
  - 适配层（`mclaw/adapters/`）：actor backend (verl DataParallelPPOActor + joint PPO+aux)、ref policy、inference engine (vLLM)、env client (agentenv)、rollout handler、logger (TensorBoard/WandB/Composite)。
  - 回归测试 (`tests/test_regressions.py`): 8 tests passing。

- Ray 集成（`feature/ray-integration` 分支）：
  - `MClawTreeRollout`：作为 verl `vLLMRollout` 的替代品，嵌入 AgentGym-RL 的 Ray 框架。
  - 并行环境交互：`ThreadPoolExecutor` 并行化 env reset / 分支执行。
  - 左 padding DataProto 布局：对齐 verl 的 `[:, -response_length-1:-1]` 切片约定。
  - `alignment` 参数：消除 signal 长度匹配歧义。

## 仓库结构

```text
MClaw/
├── README.md
├── plan.md
├── integration_guide.md
├── pyproject.toml
├── examples/
│   ├── README.md
│   ├── run_textcraft_train.sh        # 单机独立训练
│   └── run_textcraft_ray_train.sh    # Ray 分布式训练
├── tests/
│   └── test_regressions.py
└── mclaw/
    ├── __init__.py
    ├── adapters/          # verl/agentenv 适配层
    ├── clustering/
    ├── config/
    ├── core/
    ├── critic/
    ├── trainer/
    └── utils/
```

## 模块概览

- `plan.md`
  - 算法设计文档，当前实现大体按这里的模块边界推进。

- `mclaw/config/`
  - dataclass 配置定义和默认 YAML。
  - 已导出 `DEFAULT_CONFIG_PATH`、`TreeRolloutConfig`、`ClusteringConfig`、`QCriticConfig`、`AuxLossConfig`、`MClawTrainerConfig`。

- `mclaw/core/`
  - 本地批结构和外部协议。
  - `TreeNode` 现在是纯树节点定义；训练样本容器已经移到 `contracts.py`。
  - `TreeRolloutOutput` 只保留 `actor_data`、`aux_actor_data`、`critic_data` 和 `roots`，不再重复暴露原始 sample 列表。
  - `EnvironmentClientProtocol.step()` 现在统一要求返回 gym 风格 4-tuple。

- `mclaw/clustering/`
  - `BaseClusterer` 已包含实际的特征清洗、可选 PCA、确定性 K-Means 和 representative 选择。
  - 各具体 clusterer 已有特征提取实现，而不是占位文件。

- `mclaw/critic/`
  - `QHead` 已实现两层 MLP + zero-init 输出层。
  - `QCritic` 已实现批量编码、TD target 构造和 Q-head 更新。
  - `advantage.py` 已实现 executed-node 的 TD/advantage 回填和状态值估计。

- `mclaw/adapters/`
  - `VerlActorBackend`：对接 verl `DataParallelPPOActor`，支持 joint PPO + auxiliary loss。
  - `VerlReferencePolicy`：ref log prob 计算。
  - `VerlInferenceEngine`：vLLM LLM 推理。
  - `AgentEnvClient`：agentenv 环境适配。
  - `VerlRolloutHandler`：rollout sharding 管理（含 `clone()` 支持分支 fork）。
  - `StandardLogger` + `build_tracker()`：TensorBoard / WandB / Composite tracker。
  - `DataProtoAdapter`：本地批结构 ↔ verl DataProto 转换。

- `mclaw/trainer/`
  - `train_step()`、`update_actor()`、`update_q_head()` 已实现。
  - 训练后端协议位于 `trainer/contracts.py`。
  - `main.py`：`load_config()` + `build_trainer()` + `main()` 入口已完成。

- `mclaw/utils/`
  - `EmbeddingMatrixCache`、`extract_topk_logprobs()`、`build_output_grad_features()` 已实现。
  - 面向 vLLM top-k logprob 输出的解析逻辑已从“猜格式”收紧为较严格的契约。

## 当前训练数据流

1. `TreeRollout.generate_tree_rollout()` 对每个 prompt 构建 root。
2. 根节点用“重复 prompt 组成 batch”的方式生成 `root_budget` 个候选动作。
3. `QCritic.score_actions()` 对所有候选 `(state, action)` 批量打分。
4. `clusterer.cluster_candidates()` 做聚类，`BranchSelector` 选代表动作。
5. 被选动作送入环境执行，生成 observation tokens 并写回节点。
6. `compute_tree_advantage()` 反向计算 executed nodes 的 `td_target`、`q_value`、`advantage`。
7. rollout 构造三类本地 batch：
   - `ActorBatch`
   - `AuxiliaryBatch`
   - `CriticBatch`
8. `MClawTrainer.train_step()` 调用：
   - `compute_old_log_probs()`
   - `compute_ref_log_probs()`
   - 多 epoch `update_policy()`
   - 多次 `q_critic.update()`

### Ray 分布式训练数据流

当通过 AgentGym-RL 的 Ray 框架运行时，数据流变为：

1. `RayPPOTrainer.fit()` 调度 `actor_rollout_wg.generate_sequences()`。
2. Ray 将 batch 通过 `DP_COMPUTE_PROTO` 分发到各 GPU worker。
3. 每个 `ActorRolloutRefWorker` 内部调用 `MClawTreeRollout.generate_sequences()`。
4. `TreeRollout` 执行树搜索，`ThreadPoolExecutor` 并行环境交互。
5. 结果通过 `DataProtoAdapter` 转换为 `DataProto`（左 padding 布局）。
6. 回到 driver 后执行 `compute_log_prob` → `compute_advantage(mclaw)` → `update_actor`。
7. `VerlActorBackend` 联合执行 PPO + auxiliary loss 更新。

启动命令：`bash examples/run_textcraft_ray_train.sh`

## 与 AgentGym-RL 的关系

MClaw 的目标仍然是语义兼容，而不是源码耦合。

- `core/contracts.py` 定义 rollout 侧本地批结构和协议。
- `trainer/contracts.py` 定义训练后端协议。
- `MClawTrainer` 通过 `adapt_actor_batch()` / `adapt_critic_batch()` 等钩子为外部后端预留适配入口。

当前仓库可以单独迭代核心逻辑。接入 AgentGym-RL 有两种模式：
- **独立模式**（`run_textcraft_train.sh`）：MClaw 自带训练循环，单进程，适合调试。
- **Ray 模式**（`run_textcraft_ray_train.sh`）：`MClawTreeRollout` 作为 verl rollout worker 运行在 Ray 框架内，支持多 GPU 分布式训练。

## 入口和示例

- `pyproject.toml` 已注册 `mclaw-train=mclaw.trainer.main:main`
- `examples/textcraft_train.sh` 展示了目标 CLI 形状
- `integration_guide.md` 提供 15 步联调指南（含正确的 CLI override 语法 `mclaw.` 前缀）
