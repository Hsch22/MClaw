# clustering

`clustering/` 负责从候选动作中提取特征并执行聚类。

## 当前文件

- `__init__.py`：导出聚类模块公共接口。
- `base.py`：定义 `BaseClusterer` 和 `ClusterResult`。
- `hidden_state.py`：主方案接口，基于 hidden state 聚类。
- `output_grad.py`：fallback 方案接口，基于 output-layer 梯度近似。
- `logprob.py`：baseline 方案接口，用于先跑通端到端流程。
- `logit_distribution.py`：可选方案接口，偏分析用途。

## 关键职责

- 提取聚类特征
- 在根节点执行全局聚类
- 在后续步骤执行分支内聚类
- 返回聚类标签和代表动作索引

## 当前状态

- 聚类器基类和各方案文件都已落位。
- 具体特征提取、PCA、聚类和代表选择逻辑还未实现。
