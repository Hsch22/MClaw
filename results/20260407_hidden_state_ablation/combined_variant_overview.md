# Hidden-State Ablation Overview

## Result Roots

- old root: `/mnt/kangshijia/husicheng/MClaw_runtime/outputs/hidden_state_ablation_20260407_151231`
- remaining root: `/mnt/kangshijia/husicheng/MClaw_runtime/outputs/hidden_state_ablation_remaining_20260407_184641`

## Root Summaries

- old summary: `/mnt/kangshijia/husicheng/MClaw_runtime/outputs/hidden_state_ablation_20260407_151231/variant_summary.md`
- remaining summary: `/mnt/kangshijia/husicheng/MClaw_runtime/outputs/hidden_state_ablation_remaining_20260407_184641/variant_summary.md`

## Combined Variant Table

`-` means that the task-level hidden-state metric was not present in the recorded cross-task metrics for that variant.

| variant | avg_mean_max_cluster | textcraft | babyai | maze | weather | root |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| lneg1_mean | 67.10 | 46.79 | 44.42 | 56.33 | 120.88 | `hidden_state_ablation_remaining_20260407_184641` |
| lneg2_mean | 69.60 | 54.00 | 46.25 | 55.00 | 123.17 | `hidden_state_ablation_remaining_20260407_184641` |
| lneg1_lastk8 | 87.11 | 55.17 | 57.38 | 99.78 | 136.12 | `hidden_state_ablation_remaining_20260407_184641` |
| lneg2_lastk8 | 91.80 | 63.54 | 67.25 | 96.39 | 140.00 | `hidden_state_ablation_remaining_20260407_184641` |
| lneg4_last | 99.56 | - | 88.21 | 85.94 | 124.54 | `hidden_state_ablation_remaining_20260407_184641` |
| lneg1_last | 100.53 | 145.54 | 84.75 | 84.33 | 87.50 | `hidden_state_ablation_20260407_151231` |
| lneg2_last | 101.95 | - | 88.83 | 93.11 | 123.92 | `hidden_state_ablation_20260407_151231` |
| lneg1_lastk4 | 102.30 | 101.00 | 73.83 | 98.28 | 136.08 | `hidden_state_ablation_remaining_20260407_184641` |
| lneg2_lastk4 | 103.86 | 102.33 | 71.83 | 105.11 | 136.17 | `hidden_state_ablation_remaining_20260407_184641` |
| lneg8_last | 106.68 | 135.50 | 77.86 | - | - | `hidden_state_ablation_remaining_20260407_184641` |
