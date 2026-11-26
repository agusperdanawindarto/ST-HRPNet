# Experiment archive and audit notes (2026-11-27)

Historical V1-V5 artifacts are preserved, not a new validated benchmark. Final comparison: outputs/TAAC_reexperiment_v3/tables/FINAL_clean_ablation_with_mixedpatch.csv. Validation-selected thresholds and fixed 0.5 results are retained. The final notebook uses seed 42.

The chronological sequence split is 97/21/21. Saved metadata shows overlapping input timestamps at both split boundaries (16:30-16:50 and 20:00-20:20). Targets are JMA-derived HRP pseudo-labels, not measured rainfall. Independent precipitation validation was not performed. The notebook retains a historical warning about positive prevalence versus an earlier manuscript; the final paper reports the current 0.004272 test prevalence. This does not resolve dataset provenance or independence. No claim of independent event-level or multi-seed validation is made.

Raw satellite products, dataset tensor checkpoints and duplicate archives are not distributed: redistribution rights have not been verified and some arrays exceed GitHub file limits. Split and patch metadata, model weights, metrics, prediction probabilities, figures and notebooks are archived separately. Original notebooks retain local paths; configure them for your environment. Historical author and submission metadata requires author confirmation.
