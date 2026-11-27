# ST-HRPNet V6: frame-disjoint GPU pilot (2026-11-27)

Current manuscript working title: **Mixed-Sampling and False-Positive-Aware Spatiotemporal Learning for Himawari-Derived Heavy Rainfall Potential Segmentation over Indonesia**. Target journal: Journal of the Indian Society of Remote Sensing; not a claim of submission or acceptance.

## Protocol and provenance

This is a supplementary exploratory pilot, not a replacement for the historical V5 results or a completed comparative study. The V5 archive contained 97/21/21 sequences. V6 excludes six training windows crossing missing ten-minute frames and three windows at each adjacent split boundary. The retained 91/18/18 sequences have no shared time codes or identical processed-frame SHA-256 hashes across partitions. Event/date independence and independent rainfall validation remain unverified. The test set comprises 1,179,648 pixels, including 5,111 HRP pseudo-label positives (0.433265%).

A uses random patches and WBCE–Dice; D uses 50% random and 50% positive-centered patches with FP-aware loss. Both have 922,581 parameters and the same separable-convolution/ConvLSTM backbone. Seed 42; 1,800 training patches and 448 common validation patches; Adam 1e-4; global batch 8; maximum 40 epochs; patience 10. GPU run uses two Tesla T4 devices, mixed float16 with float32 recurrent/output computation, per-example loss and common validation checkpoint selection. These are deliberate V6 protocol changes, so performance differences from V5 cannot be causally attributed to removal of overlap alone.

## Verified pilot results

| Variant | Threshold | IoU | Dice/F1 | Precision | Recall | MCC | Average precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 0.82 (validation) | 0.244325 | 0.392703 | 0.437997 | 0.355899 | 0.392455 | 0.406195 |
| D | 0.61 (validation) | 0.283488 | 0.441746 | 0.475011 | 0.412835 | 0.440578 | 0.304362 |
| A | 0.50 (fixed) | 0.261352 | 0.414399 | 0.362160 | 0.484250 | 0.415875 | 0.406195 |
| D | 0.50 (fixed) | 0.290038 | 0.449658 | 0.463182 | 0.436901 | 0.447526 | 0.304362 |

D improves overlap at both operating points, but has lower average precision and, at the fixed threshold, lower recall. The selected operating point reduces D's IoU relative to its fixed threshold. These observations do not establish universal superiority or statistical significance. Average precision is the scikit-learn non-interpolated metric, not necessarily equivalent to trapezoidal PR-AUC.

[Metrics](pilot/pilot_metrics.csv) · [Split audit](pilot/split_audit.json) · [Independent local recomputation](pilot/LOCAL_VERIFICATION.json) · [Comparison panel](pilot/comparison.png) · [Executed code](source/st-hrpnet-v6-gpu-pilot.py)

## Status

Two runs complete (A/D seed 42). A/D seeds 123 and 2025 were submitted as a separate GPU batch; results are not included until independently verified. Remaining B/C/E, spatial/temporal comparators and independent precipitation validation are not completed. HRP is a satellite-display pseudo-label, not measured surface rainfall.

Original sequence tensors and target masks remain in the private research archive; only derived predictions, weights, result tables, logs/metadata and illustrations are included here. The source dataset on Kaggle is private. Reproduction requires authorized access to the archived input. The historical V1–V5 archive remains unchanged.
