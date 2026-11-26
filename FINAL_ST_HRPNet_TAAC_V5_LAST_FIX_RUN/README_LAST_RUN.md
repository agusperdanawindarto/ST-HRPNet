# FINAL LAST RUN V5

Masalah pada hasil sebelumnya:
A/B/C tidak ikut dibandingkan karena bobot lama masih memakai BASE_FILTERS=16, sedangkan model benar memakai BASE_FILTERS=24.

Solusi:
Notebook ini menambahkan Cell 12A untuk menghapus bobot/caches A/B/C lama.

Urutan jalan:
1. Jalankan notebook sampai Cell 11.
2. Jalankan Cell 12A.
3. Jalankan Cell 12 FAIR ABLATION STUDY.
4. Jalankan Cell 12B FINAL MIXEDPATCH.
5. Buka:
   outputs/TAAC_reexperiment_v3/FINAL_TITLE_DECISION_README.txt

Yang perlu dilihat:
- FINAL_clean_ablation_with_mixedpatch.csv harus berisi A, B, C, D.
- Jika A/B/C masih skipped/failed, jangan pakai hasil itu.