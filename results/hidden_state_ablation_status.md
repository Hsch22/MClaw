# Hidden-State Ablation Status

检查目录:

- `/mnt/kangshijia/husicheng/MClaw/outputs/hidden_state_ablation_20260405_160617`
- `/mnt/kangshijia/husicheng/MClaw/outputs/hidden_state_ablation_20260405_163715`
- `/mnt/kangshijia/husicheng/MClaw/outputs/hidden_state_ablation_20260405_165102`
- `/mnt/kangshijia/husicheng/MClaw/outputs/hidden_state_ablation_20260405_165509`

结论: 这些 ablation 目录都不是完整结果集，因此没有作为正式实验结果一并提交。

## 原因

- `160617` 和 `163715` 只包含 `lneg1_last` 的早期失败日志，没有生成任何 `root_cluster_audit.jsonl`、`cross_task_metrics.json` 或 `variant_summary.md`。
- `165102` 只启动了 `textcraft` 和 `babyai`，结果不完整。
- `165509` 是最接近完整的一次尝试，但仍在 wave1 失败。

## `165509` 的失败信号

- `orchestrator.log` 记录: `lneg1_last/textcraft finished with exit_code=1`、`lneg1_last/babyai finished with exit_code=1`，随后 `aborting after wave1 failure in lneg1_last`。
- `textcraft/audit.log` 与 `babyai/audit.log` 显示任务长时间等待 GPU 资源；GPU 条件满足后，又分别出现 `env server on port 39706 disappeared before audit start` 和 `env server on port 39707 disappeared before audit start`。

如果后续重新跑通并生成 `cross_task_metrics.json` / `variant_summary.md`，再把成功目录纳入版本控制更合适。
