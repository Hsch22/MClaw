# MClaw

MClaw 是一个面向 Agent RL 的树状 rollout 框架。核心思路是在每一步动作决策时先生成大量候选动作（candidate actions），再通过聚类剪枝减少真实环境交互，同时用 Q-critic 为未执行动作提供 value 估计，并最终回到 PPO 主训练链路。

当前仓库状态是“接口骨架已建立，具体算法实现待补充”。现有代码优先把模块边界、数据结构、配置入口和外部适配点固定下来，避免后续实现时出现大范围返工。

## 当前项目结构

```text
MClaw/
├── README.md
├── plan.md
├── setup.py
├── examples/
│   ├── README.md
│   └── textcraft_train.sh
└── mclaw/
    ├── README.md
    ├── __init__.py
    ├── clustering/
    │   ├── README.md
    │   ├── __init__.py
    │   ├── base.py
    │   ├── hidden_state.py
    │   ├── logit_distribution.py
    │   ├── logprob.py
    │   └── output_grad.py
    ├── config/
    │   ├── README.md
    │   ├── __init__.py
    │   └── mclaw_trainer.yaml
    ├── core/
    │   ├── README.md
    │   ├── __init__.py
    │   ├── branch_selector.py
    │   ├── contracts.py
    │   ├── tree_node.py
    │   └── tree_rollout.py
    ├── critic/
    │   ├── README.md
    │   ├── __init__.py
    │   ├── advantage.py
    │   ├── q_critic.py
    │   └── q_head.py
    ├── trainer/
    │   ├── README.md
    │   ├── __init__.py
    │   ├── contracts.py
    │   ├── main.py
    │   └── mclaw_trainer.py
    └── utils/
        ├── README.md
        ├── __init__.py
        └── vllm_hooks.py
```

## 术语约定

- 候选动作：模型在某一步生成、尚未在真实环境执行的动作。
- 被执行动作：同一步候选动作中最终真正送入环境执行的动作。
- 真实分支：由被执行动作串起来的真实环境交互路径。
- 同簇未执行动作：与被执行动作属于同一聚类簇、但未执行的候选动作。
- 辅助样本：由同簇未执行动作构造出的 auxiliary loss 输入。
- 本地批结构：MClaw 内部统一使用的 `TrajectoryRecord`、`ActorBatch`、`AuxiliaryBatch`、`CriticBatch`。
- 协议：MClaw 对 tokenizer、推理引擎、环境客户端、训练后端等外部对象要求的最小方法集合。
- 适配层：把本地批结构转换为外部训练后端可接受格式的桥接层。

## 各模块职责

- `plan.md`：算法设计、实现顺序和关键决策说明。
- `setup.py`：包安装入口，注册 `mclaw-train` 命令。
- `examples/textcraft_train.sh`：最小训练命令示例。

- `mclaw/config/`：配置类型和默认 YAML。
  - `__init__.py` 中定义 `TreeRolloutConfig`、`ClusteringConfig`、`QCriticConfig`、`MClawTrainerConfig` 等 dataclass。
  - `mclaw_trainer.yaml` 提供默认训练配置骨架。

- `mclaw/core/`：树状 rollout 的核心接口。
  - `tree_node.py` 定义 `TreeNode`、`AuxiliarySample`、`CriticSample`、`TreeRolloutOutput`。
  - `branch_selector.py` 定义根节点代表选择和分支内动作选择接口。
  - `tree_rollout.py` 定义 `TreeRollout` 主引擎接口。
  - `contracts.py` 定义本地批结构和外部依赖协议，如 `ActorBatch`、`CriticBatch`、`EnvironmentClientProtocol`、`RolloutHandlerProtocol`。

- `mclaw/clustering/`：聚类接口和不同候选动作特征方案。
  - `base.py`：`BaseClusterer` 和 `ClusterResult`。
  - `hidden_state.py`：主方案接口。
  - `output_grad.py`：fallback 方案接口。
  - `logprob.py`：baseline 方案接口。
  - `logit_distribution.py`：分析型方案接口。

- `mclaw/critic/`：Q-value 和 advantage 相关接口。
  - `q_head.py`：轻量 `QHead` 模块骨架。
  - `q_critic.py`：共享 backbone 的 `QCritic` 接口。
  - `advantage.py`：树结构上的 step-level advantage 接口。

- `mclaw/trainer/`：训练器和训练入口。
  - `mclaw_trainer.py`：`MClawTrainer` 主循环骨架。
  - `main.py`：命令行入口、配置加载和 trainer 组装入口。
  - `contracts.py`：与外部 actor/ref/logger 后端解耦的协议定义和适配边界。

- `mclaw/utils/`：辅助工具接口。
  - `vllm_hooks.py`：top-k logprob、embedding matrix cache、output-grad 特征构造接口。

## 当前实现状态

- 已完成：
  - 包结构、模块导出、配置骨架。
  - 树节点和训练样本的公共数据结构。
  - rollout、clustering、critic、trainer、utils 的主接口。
  - 命令行入口和示例脚本占位。
  - 无 `torch` 环境下的基础导入兼容。

- 尚未完成：
  - 具体聚类算法实现。
  - Q-head / Q-critic 前向和训练逻辑。
  - 树状 rollout 主循环。
  - actor / ref / env / logger 的真实适配层实现。
  - 与具体环境和训练后端的端到端联调。

## 技术原则

- 环境侧：默认不依赖环境原生 `fork`；固定环境实例池，当前设计为 `16` 个真实执行分支。
- Rollout 侧：第 `0` 轮从根状态生成 `256` 个候选动作，聚类后选 `16` 个执行；后续每个活跃分支每步固定生成 `16` 个候选动作。
- 聚类侧：主方案使用 `hidden_state` 聚类；fallback 使用 `output_grad`；先保证流程可跑通，再优化质量。
- 分支侧：后续只做分支内聚类；每个活跃分支每步只真实执行 `1` 个 action。
- 训练侧：PPO 主样本来自被执行的完整轨迹；同簇未执行动作通过 auxiliary loss 参与训练。
- 权重侧：auxiliary loss 采用“簇等权”，避免大簇吞掉小簇。
- Critic 侧：复用 actor backbone 的 FSDP module；默认只训练 `Q-head`，backbone 冻结。

## 与 AgentGym-RL 的关系

MClaw 的目标是与 AgentGym-RL 的训练数据流和调用阶段保持兼容，但不直接 `import` AgentGym-RL 代码，以避免强耦合。

当前策略是：

- 在 `mclaw/core/contracts.py` 中定义本地批结构，如 `ActorBatch`、`AuxiliaryBatch`、`CriticBatch`。
- 在 `mclaw/core/contracts.py` 和 `mclaw/trainer/contracts.py` 中定义最小协议，如 tokenizer、inference engine、environment client、actor backend、reference policy、logger。
- 在 `MClawTrainer` 中预留批适配接口，后续通过适配层把本地结构转换成外部训练后端需要的格式。

这意味着：

- MClaw 内部可以独立开发和测试。
- 后续接入 AgentGym-RL 时，只需要实现适配层，不需要把核心逻辑写死在其源码接口上。

## 兼容性约束

- 环境适配器目标：`agentgym-rl-qwen3`
- 模型家族目标：`/mnt/kangshijia/husicheng/AgentGym-RL/models/Qwen3-4B-Instruct-2507`
- 日志路径约定：`MClaw/log/{timestamp}.log`
- 参数传递方式：支持配置文件和命令行覆盖
- 部署形态：单机多卡

## 后续建议

- 先实现 `logprob` baseline，把树状 rollout 主流程跑通。
- 再接入 `QCritic` 和 `hidden_state` 聚类，把聚类与价值估计合并到同一条 FSDP forward 链路中。
- 最后补外部训练后端适配层，完成和 AgentGym-RL 风格训练流程的联调。
