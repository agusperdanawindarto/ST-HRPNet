# TAAC extended experiment summary

Paper title: **Himawari-derived heavy rainfall potential segmentation over Indonesia using an imbalance-aware spatiotemporal deep learning framework**

## Existing model comparison

Main model comparison CSV was not found. Run the original evaluation cell first.

## Ablation study

| model                                         | ablation_component                                     | threshold_mode   |   threshold |       iou |   dice_f1 |   precision |    recall |      mcc |   pr_auc |   predicted_positive_ratio |
|:----------------------------------------------|:-------------------------------------------------------|:-----------------|------------:|----------:|----------:|------------:|----------:|---------:|---------:|---------------------------:|
| Ablation_A_RandomPatch_WBCE_Dice_NoFP         | random patches + WBCE-Dice loss + fixed threshold      | fixed            |        0.5  | 0.261996  |  0.415209 |    0.277197 | 0.826918  | 0.476523 | 0.354111 |                 0.00861613 |
| Ablation_B_PatchBalanced_WBCE_Dice_NoFP       | patch-balanced + WBCE-Dice loss + fixed threshold      | fixed            |        0.5  | 0.0559428 |  0.105958 |    0.161924 | 0.0787421 | 0.111142 | 0.163516 |                 0.00140454 |
| Ablation_C_PatchBalanced_FPaware_Fixed05      | patch-balanced + FP-aware loss + fixed threshold       | fixed            |        0.5  | 0.135459  |  0.238598 |    0.144224 | 0.690314  | 0.311822 | 0.219009 |                 0.0138245  |
| Ablation_D_FullFramework_ConstrainedThreshold | patch-balanced + FP-aware loss + constrained threshold | constrained      |        0.98 | 0.165532  |  0.284046 |    0.215617 | 0.416101  | 0.296774 | 0.219009 |                 0.00557382 |

## Threshold sensitivity summary

Threshold sensitivity summary is not available. Run Cell 17C.

## Independent precipitation validation

Independent precipitation validation has not been run because external GPM/GSMaP/radar/rain-gauge masks were not found. Use the templates in external_precip_validation/ to add independent precipitation evidence.

## Event-based case analysis

|   sample_index |   target_time_code |      iou |   dice_f1 |   precision |   recall |      mcc |   predicted_positive_ratio |
|---------------:|-------------------:|---------:|----------:|------------:|---------:|---------:|---------------------------:|
|              9 |               2200 | 0.232143 |  0.376812 |    0.321981 | 0.454148 | 0.379847 |                 0.00492859 |
|              7 |               2140 | 0.251111 |  0.401421 |    0.335312 | 0.5      | 0.406995 |                 0.00514221 |
|             10 |               2210 | 0.202991 |  0.337478 |    0.280236 | 0.424107 | 0.342011 |                 0.00517273 |
|              6 |               2130 | 0.272941 |  0.428835 |    0.36478  | 0.520179 | 0.433329 |                 0.00485229 |
|             13 |               2240 | 0.158261 |  0.273273 |    0.204494 | 0.411765 | 0.286851 |                 0.00679016 |

## Suggested manuscript integration

- Add a new subsection titled **Ablation analysis of imbalance-aware components**.
- Add a new subsection titled **Threshold sensitivity and positive-ratio stability**.
- If independent precipitation masks are provided, add a subsection titled **Independent precipitation consistency assessment**.
- Add 2-3 event-based qualitative examples to strengthen applied meteorological interpretation.
