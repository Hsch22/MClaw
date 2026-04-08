# Cross-Task Summary

- output_root: `/mnt/kangshijia/husicheng/MClaw/outputs/root_audit_trainlike_20260405_090605`
- setup: `50 items/task`, `root_budget=256`, `root_clusters=16`, `n_envs=16`, `max_rounds=1`
- methods: `action, hidden_state, output_grad, logprob, logit_distribution`

## Overall Verdict

- fixed-K 方法整体排序：`hidden_state > output_grad > logprob >>> logit_distribution`
- `hidden_state` 最稳，`output_grad` 是强备选
- `action` 是 exact-match baseline，不应与 fixed-K 方法直接并排排名
- `logit_distribution` 在四个 task 上都明显塌成超大簇
- `maze` 有 `14` 个 root error，结论基于 `36` 个成功样本

## textcraft

- n_success: `50`
- n_root_errors: `0`

| method | mean_actual_clusters | mean_largest_cluster | mean_selected |
| --- | ---: | ---: | ---: |
| action | 256.00 | 1.00 | 16.00 |
| hidden_state | 16.00 | 127.50 | 16.00 |
| output_grad | 16.00 | 55.30 | 16.00 |
| logprob | 16.00 | 109.58 | 16.00 |
| logit_distribution | 16.00 | 241.00 | 16.00 |

- manual_review: `/mnt/kangshijia/husicheng/MClaw/outputs/root_audit_trainlike_20260405_090605/textcraft/manual_review.md`
- summary_json: `/mnt/kangshijia/husicheng/MClaw/outputs/root_audit_trainlike_20260405_090605/textcraft/summary.json`

## babyai

- n_success: `50`
- n_root_errors: `0`

| method | mean_actual_clusters | mean_largest_cluster | mean_selected |
| --- | ---: | ---: | ---: |
| action | 254.54 | 2.04 | 16.00 |
| hidden_state | 16.00 | 100.50 | 16.00 |
| output_grad | 16.00 | 62.20 | 16.00 |
| logprob | 16.00 | 109.56 | 16.00 |
| logit_distribution | 16.00 | 241.00 | 16.00 |

- manual_review: `/mnt/kangshijia/husicheng/MClaw/outputs/root_audit_trainlike_20260405_090605/babyai/manual_review.md`
- summary_json: `/mnt/kangshijia/husicheng/MClaw/outputs/root_audit_trainlike_20260405_090605/babyai/summary.json`

## maze

- n_success: `36`
- n_root_errors: `14`

| method | mean_actual_clusters | mean_largest_cluster | mean_selected |
| --- | ---: | ---: | ---: |
| action | 241.78 | 3.89 | 16.00 |
| hidden_state | 16.00 | 97.31 | 16.00 |
| output_grad | 16.00 | 45.83 | 16.00 |
| logprob | 16.00 | 145.08 | 16.00 |
| logit_distribution | 16.00 | 241.00 | 16.00 |

- manual_review: `/mnt/kangshijia/husicheng/MClaw/outputs/root_audit_trainlike_20260405_090605/maze/manual_review.md`
- summary_json: `/mnt/kangshijia/husicheng/MClaw/outputs/root_audit_trainlike_20260405_090605/maze/summary.json`

## weather

- n_success: `50`
- n_root_errors: `0`

| method | mean_actual_clusters | mean_largest_cluster | mean_selected |
| --- | ---: | ---: | ---: |
| action | 66.02 | 100.40 | 14.08 |
| hidden_state | 16.00 | 119.06 | 16.00 |
| output_grad | 16.00 | 120.56 | 16.00 |
| logprob | 16.00 | 158.82 | 16.00 |
| logit_distribution | 16.00 | 241.00 | 16.00 |

- manual_review: `/mnt/kangshijia/husicheng/MClaw/outputs/root_audit_trainlike_20260405_090605/weather/manual_review.md`
- summary_json: `/mnt/kangshijia/husicheng/MClaw/outputs/root_audit_trainlike_20260405_090605/weather/summary.json`
