# 2026-04-07 Hidden-State Ablation

来源目录:

- `/mnt/kangshijia/husicheng/MClaw_runtime/outputs/hidden_state_ablation_20260407_151231`
- `/mnt/kangshijia/husicheng/MClaw_runtime/outputs/hidden_state_ablation_remaining_20260407_184641`

这个结果集延续 `results/20260405_root_audit_trainlike/` 的思路，只保留适合长期复核和远端归档的精选文件，不直接提交体量巨大的原始审计明细。

## 保留文件

- `combined_variant_overview.md`: 跨两个 output root 的合并变体总览。
- `*/variant_summary.md`: 每个 output root 内部的变体排序总表。
- `*/variant_metrics.json`: 每个 output root 的结构化变体指标。
- `*/<variant>/cross_task_summary.md`: 单个变体的跨任务摘要。
- `*/<variant>/cross_task_metrics.json`: 单个变体的跨任务结构化指标。
- `*/<variant>/variant.txt`: 变体配置快照。
- `*/<variant>/<task>/summary.json`: 单任务结构化摘要。
- `*/<variant>/<task>/audit.command.txt`: 复现实验命令。
- `*/<variant>/<task>/audit.exit_code`: 单任务退出状态。

## 未纳入版本控制的原始文件

- `*/root_cluster_audit.jsonl`: 全量原始审计明细，体量最大，是本地归档文件。
- `*/summary.md`: 自动展开版 Markdown，和 `summary.json` 高度重复。
- `*/audit.log`、`*/session.log`: 临时运行日志，噪音高且不稳定。

## 结果范围

- 第一批 root: `lneg1_last`, `lneg2_last`
- 第二批 root: `lneg1_lastk4`, `lneg1_lastk8`, `lneg1_mean`, `lneg2_lastk4`, `lneg2_lastk8`, `lneg2_mean`, `lneg4_last`, `lneg8_last`

## 当前结论摘要

- 合并后最优的 hidden-state 变体集中在 `action_mean` 与 `last_k_mean`，而不是简单的 `last`.
- `lneg1_mean` / `lneg2_mean` 的平均 `mean_max_cluster` 最低，整体最稳。
- `lneg8_last` 与部分 `last` 系变体存在任务缺失项，因此合并表中保留 `-` 作为显式缺测标记。
