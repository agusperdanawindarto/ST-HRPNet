# ST-HRPNet V6: complete 21-run evaluation

Updated 2026-11-28. This archive reports a retrospective, single-acquisition case study of Himawari-derived HRP pseudo-label segmentation. It is not operational rainfall forecasting or independent meteorological validation. V6 removes shared processed input frames between partitions and discontinuous ten-minute windows (91/18/18 retained sequences), but does not establish independent events or dates.

## Protocol
All seven configurations use seeds 42, 123, 2025, 1800 training patches, 448 common validation patches, GPU mixed precision, batch 8, Adam 1e-4, up to 40 epochs, early stopping patience 10 and validation-only threshold selection. A--E use the same ST-HRPNet architecture. B/C share identical 65:35 balanced manifests; D/E share identical 50:50 mixed manifests. B1 is an adapted single-frame 2D U-Net; B2 is an adapted Conv3D encoder with temporal mean and 2D decoder. B1/B2 use the same sampling and FP-aware objective as D, not exact reproductions of published architectures. Neither B1 nor B2 isolates temporal context alone because the architectures also differ.

## Complete results
All 21 runs passed local numerical recomputation from prediction arrays, threshold audit and checkpoint/history checks. Comparator patch manifests match D. This is technical validation, not external validation or proof of statistical significance.

| Model | Selected IoU mean +/- sample SD | Fixed IoU | AP mean |
|---|---:|---:|---:|
| A | 0.2218 +/- 0.0520 | 0.2413 | 0.3071 |
| B | 0.3218 +/- 0.0630 | 0.3553 | 0.3984 |
| C | 0.3197 +/- 0.0536 | 0.3540 | 0.4080 |
| D | 0.3517 +/- 0.0615 | 0.3734 | 0.4394 |
| E | 0.3513 +/- 0.0608 | 0.3535 | 0.4505 |
| B1 | 0.4228 +/- 0.0801 | 0.4349 | 0.5770 |
| B2 | 0.3423 +/- 0.0417 | 0.3452 | 0.4097 |

**Interpretation.** D improves overlap over A and C on average, but D/E differ by only 0.00034 in mean selected IoU; E has higher AP. B1 outperforms D on mean selected and fixed IoU, Dice and AP and has fewer parameters. The data do not establish a benefit of temporal recurrence or FP-aware optimization over simple alternatives. Three seeds describe training variability, not independent events. AP is non-interpolated average precision, not trapezoidal PR-AUC. The historical V1--V5 experiments are not pooled with V6.

## Evidence
- [All run-level metrics](all_21_runs.csv); [mean and sample SD](all_21_mean_sd.csv); [paired seed IoU differences](paired_seed_iou_differences.csv)
- [Training efficiency](training_efficiency.csv): actual fit wall time, epochs, parameters; no inference latency claim.
- [Pilot](pilot/), [A/D repeats](ad_repeat/), [B/C/E](bce/), [B1/B2](comparators/), [source](source/).
- Each batch includes local verification, validation sweeps, weights, per-scene metrics, training histories, patch manifests and prediction probabilities. Raw input tensors and target masks remain private; dataset access requires authorization.
- [SHA-256 manifest](SHA256_manifest.csv).

## Kaggle
[pilot](https://www.kaggle.com/code/niluhwiwiksrirahayu/st-hrpnet-v6-gpu-pilot), [A/D repeat](https://www.kaggle.com/code/niluhwiwiksrirahayu/st-hrpnet-v6-gpu-ad-repeat), [B/C/E](https://www.kaggle.com/code/niluhwiwiksrirahayu/st-hrpnet-v6-gpu-bce-ablation), [comparators](https://www.kaggle.com/code/niluhwiwiksrirahayu/st-hrpnet-v6-gpu-comparators).
