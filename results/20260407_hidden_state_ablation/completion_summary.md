# Hidden-State Ablation Completion Summary

- snapshot_root: `results/20260407_hidden_state_ablation`
- source_roots:
  - `/mnt/kangshijia/husicheng/MClaw_runtime/outputs/hidden_state_ablation_20260407_151231`
  - `/mnt/kangshijia/husicheng/MClaw_runtime/outputs/hidden_state_ablation_remaining_20260407_184641`

## Included Output Roots

### `hidden_state_ablation_20260407_151231`

- variants: `lneg1_last`, `lneg2_last`
- files: `variant_summary.md`, `variant_metrics.json`, per-variant `cross_task_*`, `variant.txt`, per-task `summary.json`, `audit.command.txt`, `audit.exit_code`

### `hidden_state_ablation_remaining_20260407_184641`

- variants: `lneg1_lastk4`, `lneg1_lastk8`, `lneg1_mean`, `lneg2_lastk4`, `lneg2_lastk8`, `lneg2_mean`, `lneg4_last`, `lneg8_last`
- files: `variant_summary.md`, `variant_metrics.json`, per-variant `cross_task_*`, `variant.txt`, per-task `summary.json`, `audit.command.txt`, `audit.exit_code`

## Excluded Raw Files

- `root_cluster_audit.jsonl`
- `summary.md`
- `audit.log`
- `session.log`
