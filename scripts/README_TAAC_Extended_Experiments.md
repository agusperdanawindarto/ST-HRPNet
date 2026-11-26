# TAAC Extended Experiment Package

This package contains an extended notebook for the paper:

**Himawari-derived heavy rainfall potential segmentation over Indonesia using an imbalance-aware spatiotemporal deep learning framework**

The added cells are designed to strengthen the experiment for *Theoretical and Applied Climatology* by adding:

1. Threshold sensitivity analysis.
2. Ablation study for patch-balanced learning, false-positive-aware loss, and constrained thresholding.
3. Optional independent precipitation validation using GPM IMERG, GSMaP, radar, or rain-gauge-derived masks.
4. Event-based meteorological case visualization.
5. A manuscript-ready summary exported as `TAAC_extended_experiment_summary.md`.

## How to run

1. Open `ST_HRPNet_TAAC_Extended_Experiments.ipynb`.
2. Run the original cells from the beginning until the main evaluation is complete.
3. Run the new section: `17. TAAC extension experiments`.
4. If you have independent precipitation data, place it in `external_precip_validation/` using one of these formats:
   - `precip_test_masks_256.npy`
   - `masks_by_time/<target_time_code>.png`
   - `precip_grid.csv`
   - `netcdf/*.nc`

If no independent precipitation data are found, the validation cell will create templates and skip safely.
