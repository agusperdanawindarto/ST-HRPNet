# ============================================================
# 12A. RESET OLD A/B/C ABLATION WEIGHTS BEFORE FINAL RUN
# ============================================================

import shutil
from pathlib import Path

BASE_FILTERS = 24
LEARNING_RATE = 7e-4
RETRAIN_ABLATIONS = False
TRAIN_MISSING_ABLATIONS = True

print("BASE_FILTERS:", BASE_FILTERS)
print("LEARNING_RATE:", LEARNING_RATE)
print("MODELS_DIR:", MODELS_DIR)
print("CACHE_DIR:", CACHE_DIR)

# Hapus hanya A/B/C lama. D MixedPatch tidak dihapus.
abc_variants = [
    "A_RandomPatch_WBCE_Dice_NoFP",
    "B_PatchBalanced_WBCE_Dice_NoFP",
    "C_PatchBalanced_FPaware",
]

for variant in abc_variants:
    model_dir = MODELS_DIR / variant
    if model_dir.exists():
        shutil.rmtree(model_dir)
        print("Deleted old model dir:", model_dir)
    else:
        print("Model dir not found, OK:", model_dir)

# Hapus prediction cache A/B/C agar evaluasi tidak memakai hasil lama.
for variant in abc_variants:
    for suffix in ["validation_prob.npy", "test_prob.npy"]:
        cache_path = CACHE_DIR / f"{variant}_{suffix}"
        if cache_path.exists():
            cache_path.unlink()
            print("Deleted old cache:", cache_path)
        else:
            print("Cache not found, OK:", cache_path)

# Hapus tabel ablation dan final decision lama agar tidak terbaca ulang.
old_tables = [
    TABLES_DIR / "fair_ablation_study_results.csv",
    TABLES_DIR / "FINAL_clean_ablation_with_mixedpatch.csv",
    TABLES_DIR / "FINAL_title_and_novelty_decision.json",
]

for p in old_tables:
    if p.exists():
        p.unlink()
        print("Deleted old table:", p)
    else:
        print("Table not found, OK:", p)

old_decision = TAAC_OUTPUT_DIR / "FINAL_TITLE_DECISION_README.txt"
if old_decision.exists():
    old_decision.unlink()
    print("Deleted old decision:", old_decision)

print("=" * 80)
print("RESET SELESAI.")
print("Sekarang jalankan Cell 12 FAIR ABLATION STUDY sampai selesai.")
print("Setelah Cell 12 selesai, jalankan Cell 12B FINAL MIXEDPATCH.")
print("=" * 80)
