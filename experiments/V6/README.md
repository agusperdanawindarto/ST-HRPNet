# ST-HRPNet V6 — verified frame-disjoint GPU evaluation

Updated 27 November 2026. Working manuscript title: **Mixed-Sampling and False-Positive-Aware Spatiotemporal Learning for Himawari-Derived Heavy Rainfall Potential Segmentation over Indonesia**.

## Protocol and provenance

V6 retains 91/18/18 sequences from the historical 97/21/21 split. Six irregular training windows and three overlapping windows at each adjacent split boundary were excluded. Processed-frame SHA-256 and time-code checks confirm no shared frames across partitions. The test set includes 1,179,648 pixels, of which 5,111 are positive HRP pseudo-label pixels. Independent event/date generalization and independent precipitation validation are not established.

A uses random sampling with WBCE + Dice; D uses 50:50 random/positive-centered sampling with FP-aware loss. Both share 922,581 parameters, 1,800 training patches, 448 common validation patches, batch 8, Adam 1e-4, maximum 40 epochs, patience 10, and per-example loss. Seed-specific patch RNG uses seed + 101; common validation seed 244. Executed on Kaggle two-GPU Tesla T4 runtime, mixed float16 with float32 recurrence and output. Validation checkpoint selection uses foreground IoU at 0.50. Decision thresholds are selected from 91 validation candidates (0.05 to 0.95). Test evaluation uses frozen checkpoints and both selected and fixed 0.50 thresholds.

## A/D multi-seed results

### Validation-selected thresholds

| Metric | A: random | D: proposed |
|---|---:|---:|
| IoU | 0.2218 ± 0.0520 | 0.3517 ± 0.0615 |
| Dice/F1 | 0.3610 ± 0.0714 | 0.5183 ± 0.0686 |
| Precision | 0.3473 ± 0.0890 | 0.5429 ± 0.0780 |
| Recall | 0.3895 ± 0.1087 | 0.5058 ± 0.1140 |
| MCC | 0.3617 ± 0.0746 | 0.5193 ± 0.0701 |
| Average precision (AP) | 0.3071 ± 0.1003 | 0.4394 ± 0.1175 |

### Fixed threshold 0.50

| Metric | A: random | D: proposed |
|---|---:|---:|
| IoU | 0.2413 ± 0.0381 | 0.3734 ± 0.0725 |
| Dice/F1 | 0.3878 ± 0.0504 | 0.5410 ± 0.0794 |
| Precision | 0.2945 ± 0.0621 | 0.5206 ± 0.0645 |
| Recall | 0.6109 ± 0.1846 | 0.5696 ± 0.1217 |
| MCC | 0.4150 ± 0.0633 | 0.5410 ± 0.0810 |
| Average precision (AP) | 0.3071 ± 0.1003 | 0.4394 ± 0.1175 |

### Seed-level results (validation-selected)

| Seed | A threshold | D threshold | A IoU | D IoU | A AP | D AP |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 0.82 | 0.61 | 0.2443 | 0.2835 | 0.4062 | 0.3044 |
| 123 | 0.54 | 0.74 | 0.2587 | 0.3687 | 0.3095 | 0.5178 |
| 2025 | 0.51 | 0.60 | 0.1623 | 0.4028 | 0.2056 | 0.4961 |

D improves IoU at both operating points for all three seeds; AP is lower for D at seed 42, and mean recall at 0.50 is lower. Report mean ± sample SD descriptively; three seeds and correlated temporal windows are insufficient for strong significance or generalization claims. AP is not trapezoidal PR-AUC.

## Artifact index

| Artifact | Seed 42 | Seeds 123 and 2025 |
|---|---|---|
| Metrics | [CSV](pilot/pilot_metrics.csv) | [CSV](ad_repeat/pilot_metrics.csv) |
| Split audit | [JSON](pilot/split_audit.json) | [JSON](ad_repeat/split_audit.json) |
| Local recomputation | [JSON](pilot/LOCAL_VERIFICATION.json) | [JSON](ad_repeat/LOCAL_VERIFICATION.json) |
| Completion | [JSON](pilot/PILOT_COMPLETE.json) | [JSON](ad_repeat/PILOT_COMPLETE.json) |
| Qualitative comparison | [PNG](pilot/comparison.png) | [PNG](ad_repeat/comparison.png) |
| Executed source | [Python](source/st-hrpnet-v6-gpu-pilot.py) | [Python](source/ad_repeat_st-hrpnet-v6-gpu-ad-repeat.py) |

Combined outputs: [per-seed metrics](ad_all_seeds.csv), [mean and SD](ad_mean_sd.csv). Per-run directories contain best weights, training histories, per-scene CSVs, patch manifests, validation sweeps and probability arrays. The local audit recomputed metrics from predictions and verified validation-selected thresholds and split consistency. Original input tensors and masks are private.

## Remaining work

B/C/E (nine runs) are submitted and active, not yet verified. B/C share balanced sampling (65% positive-centered with minimum 16 positive pixels and jitter; 35% low-positive patches, maximum ratio 0.0005, with logged fallbacks); E matches D's 50:50 sampler without FP-aware loss. B1/B2 (six planned runs) remain pending. No independent meteorological validation has been completed. V1–V5 remain historical and are not pooled with these results.

Kaggle notebooks: [pilot](https://www.kaggle.com/code/niluhwiwiksrirahayu/st-hrpnet-v6-gpu-pilot), [repeat](https://www.kaggle.com/code/niluhwiwiksrirahayu/st-hrpnet-v6-gpu-ad-repeat), [BCE ablation](https://www.kaggle.com/code/niluhwiwiksrirahayu/st-hrpnet-v6-gpu-bce-ablation). Access requires dataset/kernel authorization.
