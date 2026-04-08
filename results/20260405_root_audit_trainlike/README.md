# 2026-04-05 Train-Like Root Audit

来源目录: `/mnt/kangshijia/husicheng/MClaw/outputs/root_audit_trainlike_20260405_090605`

这个结果集保留了足够复核结论、理解配置和追踪异常的文件，同时避免把体量过大的原始明细与运行日志直接提交到仓库。

## 保留文件

- `cross_task_summary.md`: 跨任务结论与总览表。
- `cross_task_metrics.json`: 跨任务结构化指标，便于后续脚本处理。
- `completion_summary.md`: 整体完成状态。
- `*/summary.json`: 每个 task 的结构化摘要、覆盖配置、root error 统计。
- `*/manual_review.md`: 每个 task 的代表性人工复核样本。
- `*/audit.command.txt`: 复现实验命令。
- `*/audit.exit_code`: 每个 task 的退出状态。

## 未纳入版本控制的原始文件

- `*/root_cluster_audit.jsonl`: 四个 task 合计约 688 MB，属于全量原始明细，适合本地归档，不适合直接入库。
- `*/summary.md`: 自动生成但体量很大，主要是 `jsonl` 的展开副本，信息与 `summary.json + manual_review.md` 高度重叠。
- `*.log` 与 `env.log`: 临时运行日志，噪音高且稳定性差。

## 当前结论摘要

- fixed-K 方法整体排序: `hidden_state > output_grad > logprob >>> logit_distribution`
- `hidden_state` 最稳，`output_grad` 是强备选。
- `maze` 有 `14` 个 root error，结论基于 `36` 个成功样本。

更细的指标与样例见同目录下各文件。
