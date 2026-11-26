# ============================================================
# 12B. FINAL MIXEDPATCH ABLATION AND TITLE DECISION
# ============================================================

# Jalankan cell ini setelah:
# Cell 5 dataset integrity
# Cell 6 metrics
# Cell 7 exact architecture
# Cell 8B load st_model
# Cell 9 main model evaluation
# Cell 11 ablation helpers
# Cell 12 fair ablation, jika A/B/C sudah ada.
#
# Jika Cell 12 belum dijalankan, cell ini tetap bisa melatih D,
# tetapi keputusan final paling kuat jika A/B/C sudah tersedia di fair_ablation_study_results.csv.

import gc
import json
from pathlib import Path

# ------------------------------------------------------------
# 12B.1 Safety guard
# ------------------------------------------------------------

BASE_FILTERS = 24
LEARNING_RATE = 7e-4

required_vars_12b = [
    "X_train", "y_train", "X_val", "y_val", "X_test", "y_test",
    "build_st_hrpnet_v1_patchbalanced",
    "compile_model_with_loss",
    "make_dataset",
    "evaluate_model_fair",
    "binary_metrics_np",
    "pr_auc_score_np",
]

missing_12b = [v for v in required_vars_12b if v not in globals()]
if missing_12b:
    raise RuntimeError(
        "Belum bisa menjalankan 12B. Variabel/fungsi berikut belum tersedia: "
        + ", ".join(missing_12b)
        + ". Jalankan cell 5, 6, 7, 8B, 9, dan 11 terlebih dahulu."
    )

# Pastikan arsitektur builder adalah versi exact, bukan fallback lama.
probe_model = build_st_hrpnet_v1_patchbalanced(
    seq_len=SEQ_LEN,
    n_channels=len(INPUT_PRODUCTS),
    base=BASE_FILTERS,
)
probe_layer_names = [layer.name for layer in probe_model.layers]
if ("enc1_sepconv1" not in probe_layer_names) or ("hrp_mask" not in probe_layer_names):
    raise RuntimeError(
        "Builder ST-HRPNet belum exact. Jalankan ulang Cell 7 EXACT ST-HRPNET BUILDER, "
        "pastikan BASE_FILTERS=24, lalu ulangi Cell 12B."
    )
del probe_model
gc.collect()

try:
    tf.keras.backend.clear_session()
except Exception:
    pass

# ------------------------------------------------------------
# 12B.2 Ensure main metrics exist
# ------------------------------------------------------------

if ("main_metrics" not in globals()) or ("main_metrics_05" not in globals()):
    print("main_metrics belum tersedia. Membuat evaluasi model utama terlebih dahulu...")

    if "st_model" not in globals():
        raise RuntimeError(
            "st_model belum tersedia. Jalankan Cell 8B FORCE LOAD ST_MODEL sebelum 12B."
        )

    main_metrics, main_metrics_05, main_th_df, st_val_prob, st_test_prob = evaluate_model_fair(
        st_model,
        "Full_STHRPNet_PatchBalanced_FPaware",
        input_type="sequence",
        force_predict=False,
    )

    main_eval_df = pd.DataFrame([main_metrics_05, main_metrics])
    main_eval_path = TABLES_DIR / "main_model_re_evaluation_fair_threshold.csv"
    main_eval_df.to_csv(main_eval_path, index=False)
    print("Main metrics created:", main_eval_path)
else:
    print("main_metrics sudah tersedia.")

# ------------------------------------------------------------
# 12B.3 Mixed patch sampler
# ------------------------------------------------------------

def make_mixed_patches(
    X_arr,
    y_arr,
    n_patches,
    patch_size=PATCH_SIZE,
    positive_fraction=0.50,
    min_positive_pixels=1,
    seed=SEED,
):
    """
    MixedPatch:
    - sebagian patch random untuk menjaga distribusi background alami;
    - sebagian patch dipusatkan pada piksel HRP positif agar rare pattern tetap dipelajari.
    """
    rng = np.random.default_rng(seed)

    X_arr = np.asarray(X_arr, dtype=np.float32)
    y_arr = np.asarray(y_arr, dtype=np.float32)

    n, t, h, w, c = X_arr.shape
    n_patches = int(n_patches)
    n_positive = int(round(n_patches * positive_fraction))
    n_random = n_patches - n_positive

    Xp, yp = [], []
    patch_records = []

    def append_patch(idx, top, left, patch_type):
        x_patch = X_arr[idx, :, top:top + patch_size, left:left + patch_size, :]
        y_patch = y_arr[idx, top:top + patch_size, left:left + patch_size, :]
        Xp.append(x_patch)
        yp.append(y_patch)
        patch_records.append({
            "source_index": int(idx),
            "top": int(top),
            "left": int(left),
            "patch_type": patch_type,
            "positive_ratio": float(np.mean(y_patch)),
            "positive_pixels": int(np.count_nonzero(y_patch >= 0.5)),
        })

    # 50% random patches
    for _ in tqdm(range(n_random), desc="MixedPatch random"):
        idx = int(rng.integers(0, n))
        top = int(rng.integers(0, h - patch_size + 1))
        left = int(rng.integers(0, w - patch_size + 1))
        append_patch(idx, top, left, "random")

    # 50% positive-centered patches
    positive_indices = []
    for idx in range(n):
        mask = y_arr[idx, ..., 0] if y_arr.ndim == 4 else y_arr[idx]
        if int(np.count_nonzero(mask >= 0.5)) >= int(min_positive_pixels):
            positive_indices.append(idx)

    if len(positive_indices) == 0:
        print("WARNING: Tidak ada mask positif. Positive-centered diganti random.")
        for _ in tqdm(range(n_positive), desc="MixedPatch positive fallback-random"):
            idx = int(rng.integers(0, n))
            top = int(rng.integers(0, h - patch_size + 1))
            left = int(rng.integers(0, w - patch_size + 1))
            append_patch(idx, top, left, "positive_fallback_random")
    else:
        for _ in tqdm(range(n_positive), desc="MixedPatch positive-centered"):
            idx = int(rng.choice(positive_indices))
            mask = y_arr[idx, ..., 0] if y_arr.ndim == 4 else y_arr[idx]
            coords = np.argwhere(mask >= 0.5)

            # Pilih salah satu piksel HRP sebagai anchor.
            cy, cx = coords[int(rng.integers(0, len(coords)))]

            # Jitter agar tidak terlalu selalu berada tepat di tengah.
            jitter_y = int(rng.integers(-patch_size // 4, patch_size // 4 + 1))
            jitter_x = int(rng.integers(-patch_size // 4, patch_size // 4 + 1))

            top = int(np.clip(cy - patch_size // 2 + jitter_y, 0, h - patch_size))
            left = int(np.clip(cx - patch_size // 2 + jitter_x, 0, w - patch_size))

            append_patch(idx, top, left, "positive_centered")

    Xp = np.asarray(Xp, dtype=np.float32)
    yp = np.asarray(yp, dtype=np.float32)
    meta_patch = pd.DataFrame(patch_records)

    # Shuffle final mixed patches
    order = rng.permutation(len(Xp))
    Xp = Xp[order]
    yp = yp[order]
    meta_patch = meta_patch.iloc[order].reset_index(drop=True)

    return Xp, yp, meta_patch


# ------------------------------------------------------------
# 12B.4 Build/load MixedPatch dataset
# ------------------------------------------------------------

D_VARIANT_NAME = "D_MixedPatch50Random50Positive_FPaware"
D_POSITIVE_FRACTION = 0.50
D_EPOCHS = 50
D_PATIENCE = 12

# Jumlah patch disamakan dengan patch-balanced kalau tersedia.
if "X_train_patch" in globals():
    N_TRAIN_MIXED = len(X_train_patch)
else:
    N_TRAIN_MIXED = int(globals().get("N_TRAIN_PATCHES", 1800))

if "X_val_patch" in globals():
    N_VAL_MIXED = len(X_val_patch)
else:
    N_VAL_MIXED = int(globals().get("N_VAL_PATCHES", 450))

mixed_cache = CHECKPOINT_DIR / "mixedpatch_v3"
mixed_cache.mkdir(parents=True, exist_ok=True)

X_train_mixed_path = mixed_cache / "X_train_mixed_patch.npy"
y_train_mixed_path = mixed_cache / "y_train_mixed_patch.npy"
X_val_mixed_path = mixed_cache / "X_val_mixed_patch.npy"
y_val_mixed_path = mixed_cache / "y_val_mixed_patch.npy"
meta_train_mixed_path = mixed_cache / "meta_train_mixed_patch.csv"
meta_val_mixed_path = mixed_cache / "meta_val_mixed_patch.csv"

if all(p.exists() for p in [X_train_mixed_path, y_train_mixed_path, X_val_mixed_path, y_val_mixed_path]):
    print("Loading existing MixedPatch cache...")
    X_train_mixed_patch = np.load(X_train_mixed_path)
    y_train_mixed_patch = np.load(y_train_mixed_path)
    X_val_mixed_patch = np.load(X_val_mixed_path)
    y_val_mixed_patch = np.load(y_val_mixed_path)

    meta_train_mixed = pd.read_csv(meta_train_mixed_path) if meta_train_mixed_path.exists() else pd.DataFrame()
    meta_val_mixed = pd.read_csv(meta_val_mixed_path) if meta_val_mixed_path.exists() else pd.DataFrame()
else:
    print("Creating final MixedPatch dataset...")
    X_train_mixed_patch, y_train_mixed_patch, meta_train_mixed = make_mixed_patches(
        X_train,
        y_train,
        n_patches=N_TRAIN_MIXED,
        patch_size=PATCH_SIZE,
        positive_fraction=D_POSITIVE_FRACTION,
        seed=SEED + 101,
    )

    X_val_mixed_patch, y_val_mixed_patch, meta_val_mixed = make_mixed_patches(
        X_val,
        y_val,
        n_patches=N_VAL_MIXED,
        patch_size=PATCH_SIZE,
        positive_fraction=D_POSITIVE_FRACTION,
        seed=SEED + 202,
    )

    np.save(X_train_mixed_path, X_train_mixed_patch)
    np.save(y_train_mixed_path, y_train_mixed_patch)
    np.save(X_val_mixed_path, X_val_mixed_patch)
    np.save(y_val_mixed_path, y_val_mixed_patch)
    meta_train_mixed.to_csv(meta_train_mixed_path, index=False)
    meta_val_mixed.to_csv(meta_val_mixed_path, index=False)

print("Mixed train:", X_train_mixed_patch.shape, y_train_mixed_patch.shape, "positive_ratio=", float(np.mean(y_train_mixed_patch)))
print("Mixed val  :", X_val_mixed_patch.shape, y_val_mixed_patch.shape, "positive_ratio=", float(np.mean(y_val_mixed_patch)))

mixed_summary = pd.DataFrame([
    {
        "split": "train_mixed",
        "n_patches": len(X_train_mixed_patch),
        "positive_pixel_ratio": float(np.mean(y_train_mixed_patch)),
        "positive_fraction_setting": D_POSITIVE_FRACTION,
    },
    {
        "split": "validation_mixed",
        "n_patches": len(X_val_mixed_patch),
        "positive_pixel_ratio": float(np.mean(y_val_mixed_patch)),
        "positive_fraction_setting": D_POSITIVE_FRACTION,
    },
])
mixed_summary_path = TABLES_DIR / "mixedpatch_dataset_summary.csv"
mixed_summary.to_csv(mixed_summary_path, index=False)
display(mixed_summary)
print("Saved:", mixed_summary_path)

# ------------------------------------------------------------
# 12B.5 Train/load D model
# ------------------------------------------------------------

train_mixed_ds = make_dataset(X_train_mixed_patch, y_train_mixed_patch, shuffle=True)
val_mixed_ds = make_dataset(X_val_mixed_patch, y_val_mixed_patch, shuffle=False)

D_model_dir = MODELS_DIR / D_VARIANT_NAME
D_model_dir.mkdir(parents=True, exist_ok=True)
D_weights_path = D_model_dir / f"{D_VARIANT_NAME}_best.weights.h5"

if D_weights_path.exists() and not RETRAIN_ABLATIONS:
    print("Loading existing D weights:", D_weights_path)
    model_D = build_st_hrpnet_v1_patchbalanced(
        seq_len=SEQ_LEN,
        n_channels=len(INPUT_PRODUCTS),
        base=BASE_FILTERS,
    )
    model_D = compile_model_with_loss(model_D, loss_fn="with_fp", lr=LEARNING_RATE)
    model_D.load_weights(str(D_weights_path))
else:
    print("Training final D MixedPatch model...")
    old_patience = globals().get("ABLATION_PATIENCE", 10)
    ABLATION_PATIENCE = D_PATIENCE

    model_D = build_st_hrpnet_v1_patchbalanced(
        seq_len=SEQ_LEN,
        n_channels=len(INPUT_PRODUCTS),
        base=BASE_FILTERS,
    )
    model_D = compile_model_with_loss(model_D, loss_fn="with_fp", lr=LEARNING_RATE)

    model_D, hist_D, model_dir_D = train_model_safe(
        model_D,
        train_mixed_ds,
        val_mixed_ds,
        D_VARIANT_NAME,
        epochs=D_EPOCHS,
    )

    ABLATION_PATIENCE = old_patience

print("D MixedPatch model ready.")

# ------------------------------------------------------------
# 12B.6 Evaluate D with the same full-image validation-threshold protocol
# ------------------------------------------------------------

D_metrics, D_metrics_05, D_th_df, D_val_prob, D_test_prob = evaluate_model_fair(
    model_D,
    D_VARIANT_NAME,
    input_type="sequence",
    force_predict=False,
)

D_metrics["ablation_setting"] = D_VARIANT_NAME
D_metrics["patch_strategy"] = "50% random + 50% positive-centered"
D_metrics["loss_setting"] = "WBCE + Dice + FP-aware"
D_metrics_05["ablation_setting"] = D_VARIANT_NAME + "_fixed_0_5"
D_metrics_05["patch_strategy"] = "50% random + 50% positive-centered"
D_metrics_05["loss_setting"] = "WBCE + Dice + FP-aware"

D_eval_df = pd.DataFrame([D_metrics_05, D_metrics])
D_eval_path = TABLES_DIR / "D_mixedpatch_final_results.csv"
D_eval_df.to_csv(D_eval_path, index=False)
display(D_eval_df)
print("Saved:", D_eval_path)

# ------------------------------------------------------------
# 12B.7 Merge with previous ablation table
# ------------------------------------------------------------

previous_ablation_path = TABLES_DIR / "fair_ablation_study_results.csv"
if previous_ablation_path.exists():
    previous_ablation_df = pd.read_csv(previous_ablation_path)
else:
    print("fair_ablation_study_results.csv belum ada. Membuat tabel dari main metrics saja.")
    previous_ablation_df = pd.DataFrame([main_metrics_05, main_metrics])

final_ablation_df = pd.concat([previous_ablation_df, D_eval_df], ignore_index=True)

# Drop duplicate model rows, keep latest.
if "model" in final_ablation_df.columns:
    final_ablation_df = final_ablation_df.drop_duplicates(subset=["model"], keep="last")

preferred_cols = [
    "model", "ablation_setting", "patch_strategy", "loss_setting",
    "input_type", "selected_threshold_from_validation", "threshold",
    "iou", "dice_f1", "precision", "recall", "specificity", "fpr",
    "accuracy", "balanced_accuracy", "mcc", "pr_auc",
    "actual_positive_ratio", "predicted_positive_ratio", "tp", "fp", "fn", "tn"
]
existing_cols = [c for c in preferred_cols if c in final_ablation_df.columns]
other_cols = [c for c in final_ablation_df.columns if c not in existing_cols]
final_ablation_df = final_ablation_df[existing_cols + other_cols]

final_ablation_path = TABLES_DIR / "FINAL_clean_ablation_with_mixedpatch.csv"
final_ablation_df.to_csv(final_ablation_path, index=False)
display(final_ablation_df)
print("Saved:", final_ablation_path)

# ------------------------------------------------------------
# 12B.8 Final decision and title recommendation
# ------------------------------------------------------------

# Pakai hanya row validation-selected threshold, bukan FixedThreshold.
eval_rows = final_ablation_df[
    ~final_ablation_df["model"].astype(str).str.contains("FixedThreshold", na=False)
].copy()

# Pastikan kolom numeric
for col in ["iou", "dice_f1", "mcc", "pr_auc", "predicted_positive_ratio", "actual_positive_ratio", "fpr", "recall", "precision"]:
    if col in eval_rows.columns:
        eval_rows[col] = pd.to_numeric(eval_rows[col], errors="coerce")

eval_rows = eval_rows.dropna(subset=["iou"]).sort_values("iou", ascending=False).reset_index(drop=True)

best_row = eval_rows.iloc[0].to_dict()
best_model = str(best_row.get("model", "unknown"))
best_iou = float(best_row.get("iou", np.nan))

D_row_df = eval_rows[eval_rows["model"].astype(str) == D_VARIANT_NAME]
if len(D_row_df) > 0:
    D_row = D_row_df.iloc[0].to_dict()
    D_iou = float(D_row.get("iou", np.nan))
else:
    D_row = {}
    D_iou = np.nan

A_row_df = eval_rows[eval_rows["model"].astype(str) == "A_RandomPatch_WBCE_Dice_NoFP"]
if len(A_row_df) > 0:
    A_row = A_row_df.iloc[0].to_dict()
    A_iou = float(A_row.get("iou", np.nan))
else:
    A_row = {}
    A_iou = np.nan

# Title logic
# D boleh dianggap aman jika:
# 1. D adalah terbaik, atau
# 2. D minimal 98% dari A dan lebih realistis dari sisi positive ratio/FPR.
D_close_to_A = False
if not np.isnan(D_iou) and not np.isnan(A_iou) and A_iou > 0:
    D_close_to_A = D_iou >= 0.98 * A_iou

D_is_best = (best_model == D_VARIANT_NAME)

D_ratio_ok = False
if D_row:
    ar = float(D_row.get("actual_positive_ratio", np.nan))
    pr = float(D_row.get("predicted_positive_ratio", np.nan))
    if ar > 0:
        D_ratio_ok = (0.5 * ar <= pr <= 2.0 * ar)

if D_is_best or (D_close_to_A and D_ratio_ok):
    recommended_title = (
        "Himawari-derived heavy rainfall potential segmentation over Indonesia "
        "using an imbalance-aware spatiotemporal deep learning framework"
    )
    novelty_position = (
        "AMAN: mixed imbalance-aware patch learning dapat dipertahankan sebagai komponen utama, "
        "karena D menjadi terbaik atau sangat dekat dengan random patch sambil menjaga distribusi positif lebih realistis."
    )
    paper_decision = "KEEP_IMBALANCE_AWARE_TITLE"
else:
    recommended_title = (
        "Himawari-derived heavy rainfall potential segmentation over Indonesia "
        "using a spatiotemporal deep learning framework"
    )
    novelty_position = (
        "SESUAIKAN: jangan menjadikan imbalance-aware sebagai klaim utama. "
        "Tulis imbalance-aware/mixed patch learning sebagai strategi pelatihan tambahan, "
        "sedangkan kontribusi utama adalah framework spatiotemporal untuk segmentasi HRP Himawari."
    )
    paper_decision = "ADJUST_TITLE_TO_RESULTS"

decision = {
    "paper_decision": paper_decision,
    "recommended_title": recommended_title,
    "novelty_position": novelty_position,
    "best_model_by_iou": best_model,
    "best_iou": best_iou,
    "D_model_iou": D_iou,
    "A_randompatch_iou": A_iou,
    "D_is_best": bool(D_is_best),
    "D_close_to_A_98pct": bool(D_close_to_A),
    "D_ratio_ok": bool(D_ratio_ok),
    "table_used": str(final_ablation_path),
}

decision_json_path = TABLES_DIR / "FINAL_title_and_novelty_decision.json"
with open(decision_json_path, "w", encoding="utf-8") as f:
    json.dump(decision, f, indent=2)

decision_txt_path = TAAC_OUTPUT_DIR / "FINAL_TITLE_DECISION_README.txt"
with open(decision_txt_path, "w", encoding="utf-8") as f:
    f.write("FINAL TITLE AND NOVELTY DECISION\n")
    f.write("=" * 80 + "\n\n")
    for k, v in decision.items():
        f.write(f"{k}: {v}\n")

print("=" * 90)
print("FINAL DECISION")
print("=" * 90)
print("Best model by IoU:", best_model)
print("Best IoU:", best_iou)
print("D MixedPatch IoU:", D_iou)
print("A RandomPatch IoU:", A_iou)
print("Decision:", paper_decision)
print("Recommended title:")
print(recommended_title)
print("\nNovelty position:")
print(novelty_position)
print("\nSaved:")
print("-", final_ablation_path)
print("-", decision_json_path)
print("-", decision_txt_path)
print("=" * 90)

# ------------------------------------------------------------
# 12B.9 Final plot
# ------------------------------------------------------------

if len(eval_rows) > 0:
    plt.figure(figsize=(11, 5))
    x = np.arange(len(eval_rows))
    plt.bar(x, eval_rows["iou"])
    plt.xticks(x, eval_rows["model"], rotation=35, ha="right")
    plt.ylabel("IoU")
    plt.title("Final clean ablation with MixedPatch strategy")
    plt.tight_layout()
    fig_path = FIGURES_DIR / "FINAL_clean_ablation_with_mixedpatch_iou.png"
    plt.savefig(fig_path, dpi=300)
    plt.show()
    print("Saved figure:", fig_path)

    plt.figure(figsize=(11, 5))
    plt.bar(x, eval_rows["mcc"])
    plt.xticks(x, eval_rows["model"], rotation=35, ha="right")
    plt.ylabel("MCC")
    plt.title("Final clean ablation with MixedPatch strategy")
    plt.tight_layout()
    fig_path = FIGURES_DIR / "FINAL_clean_ablation_with_mixedpatch_mcc.png"
    plt.savefig(fig_path, dpi=300)
    plt.show()
    print("Saved figure:", fig_path)
