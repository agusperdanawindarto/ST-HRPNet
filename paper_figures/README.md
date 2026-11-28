# Manuscript figure archive

Paper: *Sampling, Loss and Architecture Trade-offs in Himawari-Derived Heavy Rainfall Potential Segmentation over Indonesia*.

Figures 1–6 and 9 were extracted directly from the approved pre-LaTeX Word manuscript. Figures 7–8 were regenerated from original archived probability arrays and reference masks (same two test indices and seed, same displayed configurations and crop rule), rather than inferred from rasterized error maps. They are scientifically matched replacements, not byte-identical to the current Word images. No training was repeated. The full unpublished manuscript is not included.

## [Fig. 1 — dataset partition](Fig01_dataset_partition.png)

Fig. 1 Himawari input and HRP pseudo-label preparation, temporal sequence construction and frame-disjoint partition

## [Fig. 2 — dataset examples](Fig02_dataset_examples.png)

Fig. 2 Illustrative archived B13 display and aligned HRP pseudo-label crops (not evaluation scenes)

## [Fig. 3 — research framework](Fig03_research_framework.png)

Fig. 3 Frame-disjoint experimental framework, showing literature foundations, five internal configurations and two task-matched architecture alternatives. Solid arrows indicate experimental flow; dashed arrows indicate architectural or methodological context. Seven configurations are evaluated over three seeds.

## [Fig. 4 — ST HRPNet](Fig04_ST_HRPNet.png)

Fig. 4 Architecture and learning framework of ST-HRPNet (Variant D): (a) mixed patch sampling (S1), shared spatiotemporal backbone, FP-aware composite loss (S2), and validation-based checkpoint selection; (b) frozen-checkpoint evaluation with validation-only threshold selection. Blue denotes shared architectural components, amber denotes the evaluated training strategies, and purple denotes target and decision stages. Solid arrows indicate data and decision flow; dashed arrows denote skip features or parameter-update feedback.

## [Fig. 5 — benchmark](Fig05_benchmark.png)

Fig. 5 Mean IoU, Dice/F1 and AP at validation-selected thresholds; error bars denote sample standard deviation across three seeds

## [Fig. 6 — seed variability](Fig06_seed_variability.png)

Fig. 6 Seed-level test IoU at validation-selected thresholds for all seven configurations

## [Fig. 7 — qualitative masks](Fig07_qualitative_masks.png)

Fig. 7 Qualitative binary outputs for two held-out test windows, seed 42. White denotes the reference HRP mask in the first column. Cyan denotes the thresholded predicted positive mask, and the white contour over each prediction indicates the reference boundary. Parenthetical values are validation-selected thresholds.

## [Fig. 8 — error maps](Fig08_error_maps.png)

Fig. 8 Segmentation error maps for two held-out test windows, seed 42, for selected configurations A, D, E, B1 and B2. Green: true positive; magenta: false positive; orange: false negative; black: true negative. Parenthetical values are validation-selected thresholds.

## [Fig. 9 — operating point](Fig09_operating_point.png)

Fig. 9 Operating-point sensitivity across the seven configurations (means over three seeds). The fixed threshold is 0.50 and the alternative threshold is selected on the validation set without retraining. FPR is shown in units of ×10⁻³.

## Supplementary visual evidence

- [S1: Full 256 × 256 B13, reference, predictions of all seven configurations, and per-image IoU](Supplement_S1_full_image_all_models.png). Seed 42; fixed indices 0, 5, 11, 17 selected by spacing, not performance.
- [S2: Probability maps](Supplement_S2_probability_maps.png). Seed 42, indices 0 and 17, fixed probability scale 0–1.
- [S3: Mean per-scene IoU](Supplement_S3_scene_IoU.png). All 18 test windows and three seeds. Undefined IoU for empty union is recorded as NA; means ignore NA.
- [Verified scene-level CSV](per_scene_verified.csv): 378 model/seed/scene combinations. Confusion counts recomputed directly from probability arrays and validated against archived per-scene and pooled metrics.

Source: [complete experimental archive](../experiments/V6/README.md), [run-level results](../experiments/V6/all_21_runs.csv), [mean and SD](../experiments/V6/all_21_mean_sd.csv). All targets are product-derived HRP pseudo-labels, not independent rainfall observations. No raw tensors, standalone masks or credentials are published. Figures are rendered composites for research reporting; original source imagery redistribution rights remain unverified.
