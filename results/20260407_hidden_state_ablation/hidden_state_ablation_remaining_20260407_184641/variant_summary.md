# Hidden-State Ablation Summary

- output_root: `/mnt/kangshijia/husicheng/MClaw_runtime/outputs/hidden_state_ablation_remaining_20260407_184641`

| variant | avg_mean_max_cluster | textcraft | babyai | maze | weather | overrides |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| lneg1_mean | 67.10 | 46.79 | 44.42 | 56.33 | 120.88 | `mclaw.clustering.method=hidden_state mclaw.clustering.hidden_state.layer=-1 mclaw.clustering.hidden_state.token_pooling=action_mean mclaw.clustering.hidden_state.last_k=4` |
| lneg2_mean | 69.60 | 54.00 | 46.25 | 55.00 | 123.17 | `mclaw.clustering.method=hidden_state mclaw.clustering.hidden_state.layer=-2 mclaw.clustering.hidden_state.token_pooling=action_mean mclaw.clustering.hidden_state.last_k=4` |
| lneg1_lastk8 | 87.11 | 55.17 | 57.38 | 99.78 | 136.12 | `mclaw.clustering.method=hidden_state mclaw.clustering.hidden_state.layer=-1 mclaw.clustering.hidden_state.token_pooling=last_k_mean mclaw.clustering.hidden_state.last_k=8` |
| lneg2_lastk8 | 91.80 | 63.54 | 67.25 | 96.39 | 140.00 | `mclaw.clustering.method=hidden_state mclaw.clustering.hidden_state.layer=-2 mclaw.clustering.hidden_state.token_pooling=last_k_mean mclaw.clustering.hidden_state.last_k=8` |
| lneg4_last | 99.56 | - | 88.21 | 85.94 | 124.54 | `mclaw.clustering.method=hidden_state mclaw.clustering.hidden_state.layer=-4 mclaw.clustering.hidden_state.token_pooling=last mclaw.clustering.hidden_state.last_k=4` |
| lneg1_lastk4 | 102.30 | 101.00 | 73.83 | 98.28 | 136.08 | `mclaw.clustering.method=hidden_state mclaw.clustering.hidden_state.layer=-1 mclaw.clustering.hidden_state.token_pooling=last_k_mean mclaw.clustering.hidden_state.last_k=4` |
| lneg2_lastk4 | 103.86 | 102.33 | 71.83 | 105.11 | 136.17 | `mclaw.clustering.method=hidden_state mclaw.clustering.hidden_state.layer=-2 mclaw.clustering.hidden_state.token_pooling=last_k_mean mclaw.clustering.hidden_state.last_k=4` |
| lneg8_last | 106.68 | 135.50 | 77.86 | - | - | `mclaw.clustering.method=hidden_state mclaw.clustering.hidden_state.layer=-8 mclaw.clustering.hidden_state.token_pooling=last mclaw.clustering.hidden_state.last_k=4` |
