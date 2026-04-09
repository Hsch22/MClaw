# Hidden-State Ablation Summary

- output_root: `/mnt/kangshijia/husicheng/MClaw_runtime/outputs/hidden_state_ablation_20260407_151231`

| variant | avg_mean_max_cluster | textcraft | babyai | maze | weather | overrides |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| lneg1_last | 100.53 | 145.54 | 84.75 | 84.33 | 87.50 | `mclaw.clustering.method=hidden_state mclaw.clustering.hidden_state.layer=-1 mclaw.clustering.hidden_state.token_pooling=last mclaw.clustering.hidden_state.last_k=4` |
| lneg2_last | 101.95 | - | 88.83 | 93.11 | 123.92 | `mclaw.clustering.method=hidden_state mclaw.clustering.hidden_state.layer=-2 mclaw.clustering.hidden_state.token_pooling=last mclaw.clustering.hidden_state.last_k=4` |
