# ST-HRPNet TAAC Robust Re-Experiment V3

Cara pakai:

1. Extract paket ini ke:
   `D:\RISETKOLABORASI\Artikel Siap Kirim\QUANTUM\Program`

2. Buka:
   `ST_HRPNet_TAAC_Robust_ReExperiment_V3.ipynb`

3. Pastikan folder output utama lama benar:
   `D:\RISETKOLABORASI\Artikel Siap Kirim\QUANTUM\Program\outputs\ST_HRPNet_v1_Himawari_Indonesia_PATCHBALANCED_V2\jma_snapshot_20260529_0126UTC`

4. Jika `X_train`, `X_val`, `X_test`, dll belum ada:
   jalankan notebook utama sampai temporal split, lalu jalankan:
   `taac_save_array_checkpoints()`

5. Setelah checkpoint ada, notebook ini bisa dijalankan ulang tanpa preprocessing penuh.

Output akan tersimpan di:
`D:\RISETKOLABORASI\Artikel Siap Kirim\QUANTUM\Program\outputs\TAAC_reexperiment_v3`

Catatan:
- Jika `external_precip_validation/precip_test_masks_256.npy` belum ada, jangan klaim independent precipitation validation.
- Ablation baru layak masuk paper jika dataset ratio konsisten dan hasilnya tidak melemahkan novelty.