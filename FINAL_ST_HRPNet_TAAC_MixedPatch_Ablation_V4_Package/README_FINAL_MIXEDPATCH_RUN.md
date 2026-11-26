# Final MixedPatch Ablation Run

Paket ini adalah versi final untuk eksperimen ulang TAAC.

File utama:
`FINAL_ST_HRPNet_TAAC_Robust_ReExperiment_V4_MixedPatch.ipynb`

Yang berubah:
1. `BASE_FILTERS` sudah dipastikan `24`.
2. `LEARNING_RATE` sudah dipastikan `7e-4`.
3. Ditambahkan **Cell 12B Final MixedPatch Ablation and Title Decision**.

Cara jalan paling aman:
1. Buka notebook `FINAL_ST_HRPNet_TAAC_Robust_ReExperiment_V4_MixedPatch.ipynb`.
2. Jalankan berurutan sampai Cell 12.
3. Jalankan Cell 12B.
4. Lihat file keputusan:
   `outputs/TAAC_reexperiment_v3/FINAL_TITLE_DECISION_README.txt`

Output utama:
- `FINAL_clean_ablation_with_mixedpatch.csv`
- `FINAL_title_and_novelty_decision.json`
- `FINAL_TITLE_DECISION_README.txt`
- `FINAL_clean_ablation_with_mixedpatch_iou.png`
- `FINAL_clean_ablation_with_mixedpatch_mcc.png`

Aturan keputusan:
- Jika D MixedPatch terbaik atau minimal 98% dari A dan positive ratio realistis, judul tetap memakai `imbalance-aware`.
- Jika A RandomPatch tetap jauh lebih baik, judul disesuaikan menjadi `spatiotemporal deep learning framework` tanpa klaim utama imbalance-aware.