
# %% [markdown]
"""
# TAAC Robust Re-Experiment Notebook V3

Notebook ini untuk memperbaiki eksperimen sebelum penulisan ulang paper ke **Theoretical and Applied Climatology**.

Judul target:

**Himawari-derived heavy rainfall potential segmentation over Indonesia using an imbalance-aware spatiotemporal deep learning framework**

Fokus perbaikan:
1. menjaga dataset tetap konsisten dengan paper utama;
2. tidak download ulang Himawari jika file lokal sudah ada;
3. menyimpan/memuat checkpoint `X_train`, `X_val`, `X_test`, dan mask;
4. threshold sensitivity dengan metrik aman dari overflow;
5. fair ablation: semua varian memakai validation-selected constrained threshold dan full-image test evaluation;
6. external precipitation validation hanya jika data eksternal benar-benar tersedia;
7. export tabel/figure/ringkasan siap masuk paper.
"""

# %%
# ============================================================
# 0. OPTIONAL INSTALL
# ============================================================

# Jalankan hanya jika library belum tersedia.
# !pip install numpy pandas matplotlib scikit-learn opencv-python pillow tqdm tensorflow openpyxl


# %%
# ============================================================
# 1. IMPORTS, SEED, GPU
# ============================================================

import os, re, gc, json, math, time, random, warnings, shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import cv2

from sklearn.metrics import precision_recall_curve, auc

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(SEED)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
    print("GPU detected:", gpus)
else:
    print("No GPU detected. Running on CPU.")

print("Imports ready.")


# %%
# ============================================================
# 2. MAIN CONFIGURATION
# ============================================================

BASE_DIR = Path(r"D:\RISETKOLABORASI\Artikel Siap Kirim\QUANTUM\Program")

MAIN_OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "ST_HRPNet_v1_Himawari_Indonesia_PATCHBALANCED_V2"
    / "jma_snapshot_20260529_0126UTC"
)

TAAC_OUTPUT_DIR = BASE_DIR / "outputs" / "TAAC_reexperiment_v3"
TABLES_DIR = TAAC_OUTPUT_DIR / "tables"
FIGURES_DIR = TAAC_OUTPUT_DIR / "figures"
MODELS_DIR = TAAC_OUTPUT_DIR / "models"
CACHE_DIR = TAAC_OUTPUT_DIR / "cache"
CHECKPOINT_DIR = BASE_DIR / "taac_checkpoints_v3"
EXTERNAL_PRECIP_DIR = BASE_DIR / "external_precip_validation"

for d in [TAAC_OUTPUT_DIR, TABLES_DIR, FIGURES_DIR, MODELS_DIR, CACHE_DIR, CHECKPOINT_DIR, EXTERNAL_PRECIP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

INPUT_PRODUCTS = ["b13", "b08", "b07", "b03"]
TARGET_PRODUCT = "hrp"
SEQ_LEN = 4
IMG_SIZE = 256
PATCH_SIZE = 128
BATCH_SIZE = 8
BASE_FILTERS = 16

RUN_THRESHOLD_SENSITIVITY = True
RUN_FAIR_ABLATION = True
RUN_EXTERNAL_PRECIP_VALIDATION = True
RUN_EVENT_CASE_ANALYSIS = True

TRAIN_MISSING_ABLATIONS = True
RETRAIN_ABLATIONS = False
EPOCHS_ABLATION = 40
ABLATION_PATIENCE = 10

EXPECTED_TEST_POS_RATIO = 0.0253
POS_RATIO_TOLERANCE = 0.0100

THRESHOLD_VALUES = np.round(np.arange(0.05, 0.951, 0.01), 4)
RATIO_LOWER_MULT = 0.50
RATIO_UPPER_MULT = 2.00

PAPER_TITLE_TAAC = (
    "Himawari-derived heavy rainfall potential segmentation over Indonesia "
    "using an imbalance-aware spatiotemporal deep learning framework"
)

print("BASE_DIR:", BASE_DIR)
print("MAIN_OUTPUT_DIR:", MAIN_OUTPUT_DIR)
print("TAAC_OUTPUT_DIR:", TAAC_OUTPUT_DIR)


# %% [markdown]
"""
## 3. Cache-safe Himawari data handling

Cell ini opsional. Gunakan untuk mengganti cell download lama.  
Jika manifest/file lokal sudah valid, tidak ada download ulang.
"""

# %%
# ============================================================
# 3. CACHE-SAFE HIMAWARI R2W DATA HANDLING
# ============================================================

from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup

if "DATASET_TAG" not in globals():
    DATASET_TAG = "jma_snapshot_20260529_0126UTC"

if "RAW_DIR" not in globals():
    RAW_DIR = BASE_DIR / "dataset_himawari_r2w" / "raw"

if "MANIFEST_DIR" not in globals():
    MANIFEST_DIR = BASE_DIR / "dataset_himawari_r2w" / "manifests"

if "DOWNLOAD_PRODUCTS" not in globals():
    DOWNLOAD_PRODUCTS = ["b13", "b08", "b07", "b03", "hrp", "snd"]

if "MAX_TIMES" not in globals():
    MAX_TIMES = None

if "JMA_LIST_URL" not in globals():
    JMA_LIST_URL = ""

if "JMA_ROOT_URL" not in globals():
    JMA_ROOT_URL = "https://www.jma.go.jp/bosai/himawari/data/"

USE_LOCAL_CACHE = True
FORCE_PARSE_JMA_LIST = False
OVERWRITE_DOWNLOAD = False

manifest_path = MANIFEST_DIR / f"manifest_{DATASET_TAG}.csv"


def safe_get(url, timeout=60, retries=3, sleep_sec=2):
    if not url:
        raise RuntimeError("JMA_LIST_URL belum diisi. Gunakan parser dari notebook utama atau isi URL JMA yang benar.")
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            last_err = e
            print(f"[Retry {attempt}/{retries}] gagal akses {url}: {e}")
            time.sleep(sleep_sec)
    raise RuntimeError(f"Gagal akses setelah {retries} percobaan: {url}") from last_err


def is_valid_jpeg(path):
    path = Path(path)
    if (not path.exists()) or path.stat().st_size < 1024:
        return False
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return img is not None and img.size > 0


def validate_manifest_files(manifest, products):
    if manifest is None or manifest.empty:
        return False

    required_cols = ["time_code"] + list(products)
    missing_cols = [c for c in required_cols if c not in manifest.columns]
    if missing_cols:
        print("Manifest tidak lengkap. Kolom hilang:", missing_cols)
        return False

    flags = []
    for _, row in manifest.iterrows():
        ok = True
        for product in products:
            fp = Path(str(row[product]))
            if not is_valid_jpeg(fp):
                ok = False
                break
        flags.append(ok)

    print(f"Validasi manifest lokal: {sum(flags)}/{len(flags)} timestamp valid.")
    return len(flags) > 0 and sum(flags) == len(flags)


def load_existing_manifest_if_valid(manifest_path, products):
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return None
    try:
        manifest = pd.read_csv(manifest_path)
        if validate_manifest_files(manifest, products):
            print("Manifest lokal valid. Download dilewati.")
            return manifest.reset_index(drop=True)
        print("Manifest ditemukan tetapi ada file hilang/rusak.")
    except Exception as e:
        print("Gagal membaca manifest lokal:", e)
    return None


def extract_time_code_from_filename(path, product):
    name = Path(path).name.lower()
    pattern = re.compile(rf"r2w_{re.escape(product.lower())}_(\d{{4}})\.jpg$")
    m = pattern.search(name)
    return None if not m else m.group(1)


def build_local_manifest_from_raw(raw_dir, products):
    raw_dir = Path(raw_dir)
    product_maps = {}
    for product in products:
        product_dir = raw_dir / product
        product_maps[product] = {}
        if not product_dir.exists():
            continue
        for fp in sorted(product_dir.glob(f"r2w_{product}_*.jpg")):
            time_code = extract_time_code_from_filename(fp, product)
            if time_code and is_valid_jpeg(fp):
                product_maps[product][time_code] = str(fp)

    common_times = None
    for product in products:
        times = set(product_maps.get(product, {}).keys())
        common_times = times if common_times is None else common_times.intersection(times)

    if not common_times:
        print("Tidak ada timestamp lokal lengkap untuk semua produk.")
        return pd.DataFrame()

    common_times = sorted(common_times)
    if MAX_TIMES is not None:
        common_times = common_times[:int(MAX_TIMES)]

    rows = []
    for time_code in common_times:
        row = {"time_code": time_code}
        ok = True
        for product in products:
            fp = product_maps[product].get(time_code)
            row[product] = fp
            row[f"{product}_url"] = ""
            if fp is None or not is_valid_jpeg(fp):
                ok = False
        row["is_complete_and_valid"] = ok
        rows.append(row)

    manifest = pd.DataFrame(rows).sort_values("time_code").reset_index(drop=True)
    print("Manifest lokal dibangun dari RAW_DIR:", len(manifest), "timestamp.")
    return manifest


def prepare_manifest_cache_safe():
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    manifest = None
    if USE_LOCAL_CACHE and not FORCE_PARSE_JMA_LIST:
        manifest = load_existing_manifest_if_valid(manifest_path, DOWNLOAD_PRODUCTS)

    if manifest is None and USE_LOCAL_CACHE and not FORCE_PARSE_JMA_LIST:
        local_manifest = build_local_manifest_from_raw(RAW_DIR, DOWNLOAD_PRODUCTS)
        if not local_manifest.empty and validate_manifest_files(local_manifest, DOWNLOAD_PRODUCTS):
            manifest = local_manifest.copy()
            manifest.to_csv(manifest_path, index=False)
            print("Manifest baru dibuat dari file lokal:", manifest_path)

    if manifest is None:
        raise RuntimeError(
            "Data lokal belum lengkap. Untuk menghindari salah download, jalankan cell download asli dari notebook utama "
            "atau isi JMA_LIST_URL yang benar pada cell konfigurasi."
        )

    valid_manifest = manifest[manifest["is_complete_and_valid"]].reset_index(drop=True)
    if valid_manifest.empty:
        raise RuntimeError("Tidak ada file valid. Cek folder RAW_DIR/manifest.")

    print("Manifest saved/used:", manifest_path)
    print("Valid timestamps:", len(valid_manifest))
    display(valid_manifest.head())
    return valid_manifest

# Jalankan hanya jika ingin cek/membuat manifest lokal:
# valid_manifest = prepare_manifest_cache_safe()


# %% [markdown]
"""
## 4. Checkpoint data split

Bagian ini adalah kunci agar eksperimen tidak error lagi setelah kernel restart.  
Setelah notebook utama berhasil sampai temporal split, jalankan `taac_save_array_checkpoints()`.
"""

# %%
# ============================================================
# 4A. SAVE DATA SPLIT CHECKPOINT
# ============================================================

def taac_save_array_checkpoints():
    required = ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]
    missing = [v for v in required if v not in globals()]
    if missing:
        raise RuntimeError("Belum bisa save checkpoint. Variabel hilang: " + ", ".join(missing))

    for name in required:
        np.save(CHECKPOINT_DIR / f"{name}.npy", globals()[name])

    for name in ["X_train_patch", "y_train_patch", "X_val_patch", "y_val_patch"]:
        if name in globals():
            np.save(CHECKPOINT_DIR / f"{name}.npy", globals()[name])

    for name in ["meta_train", "meta_val", "meta_test"]:
        if name in globals() and isinstance(globals()[name], pd.DataFrame):
            globals()[name].to_csv(CHECKPOINT_DIR / f"{name}.csv", index=False)

    print("Array checkpoints saved to:", CHECKPOINT_DIR)

# Jalankan setelah split/patch selesai:
# taac_save_array_checkpoints()


# %%
# ============================================================
# 4B. LOAD DATA SPLIT CHECKPOINT
# ============================================================

def taac_load_array_checkpoints(required_patch=False):
    required = ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]
    missing_files = []
    for name in required:
        path = CHECKPOINT_DIR / f"{name}.npy"
        if not path.exists():
            missing_files.append(str(path))

    if missing_files:
        raise FileNotFoundError(
            "Checkpoint array belum lengkap. Jalankan notebook utama sampai split, "
            "lalu jalankan taac_save_array_checkpoints(). Missing: " + "; ".join(missing_files)
        )

    for name in required:
        globals()[name] = np.load(CHECKPOINT_DIR / f"{name}.npy")

    for name in ["X_train_patch", "y_train_patch", "X_val_patch", "y_val_patch"]:
        path = CHECKPOINT_DIR / f"{name}.npy"
        if path.exists():
            globals()[name] = np.load(path)

    for name in ["meta_train", "meta_val", "meta_test"]:
        path = CHECKPOINT_DIR / f"{name}.csv"
        if path.exists():
            globals()[name] = pd.read_csv(path)

    if required_patch:
        missing_patch = [n for n in ["X_train_patch", "y_train_patch", "X_val_patch", "y_val_patch"] if n not in globals()]
        if missing_patch:
            raise RuntimeError("Patch checkpoint belum ada: " + ", ".join(missing_patch))

    print("Array checkpoints loaded.")
    for name in required:
        print(name, globals()[name].shape, "positive_ratio=", float(np.mean(globals()[name])))

# Jika kernel restart, jalankan:
# taac_load_array_checkpoints(required_patch=False)


# %%
# ============================================================
# 4C. AUTO LOAD CHECKPOINT IF CORE VARIABLES ARE MISSING
# ============================================================

core_vars = ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]
missing_core = [v for v in core_vars if v not in globals()]

if missing_core:
    print("Core variables belum ada di memory:", missing_core)
    try:
        taac_load_array_checkpoints(required_patch=False)
    except Exception as e:
        print("Auto-load checkpoint gagal:", e)
        print("Solusi:")
        print("1. Jalankan notebook utama sampai temporal split dan patch creation.")
        print("2. Jalankan taac_save_array_checkpoints().")
        print("3. Buka ulang notebook ini dan jalankan dari cell ini.")
else:
    print("Core variables tersedia di memory.")
    for name in core_vars:
        print(name, globals()[name].shape, "positive_ratio=", float(np.mean(globals()[name])))


# %%
# ============================================================
# 5. DATASET INTEGRITY CHECK
# ============================================================

def taac_dataset_integrity_report():
    required = ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]
    missing = [v for v in required if v not in globals()]
    if missing:
        raise RuntimeError("Variabel dataset belum tersedia: " + ", ".join(missing))

    rows = []
    for split, X_arr, y_arr in [
        ("train", X_train, y_train),
        ("validation", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        rows.append({
            "split": split,
            "n_sequences": int(len(X_arr)),
            "X_shape": str(X_arr.shape),
            "y_shape": str(y_arr.shape),
            "positive_pixel_ratio": float(np.mean(y_arr)),
            "min_y": float(np.min(y_arr)),
            "max_y": float(np.max(y_arr)),
        })

    df = pd.DataFrame(rows)
    out_path = TABLES_DIR / "dataset_integrity_report.csv"
    df.to_csv(out_path, index=False)
    display(df)
    print("Saved:", out_path)

    test_ratio = float(np.mean(y_test))
    diff = abs(test_ratio - EXPECTED_TEST_POS_RATIO)

    if diff > POS_RATIO_TOLERANCE:
        print("=" * 80)
        print("WARNING: Test positive-pixel ratio tidak konsisten dengan paper utama.")
        print("Expected sekitar:", EXPECTED_TEST_POS_RATIO)
        print("Current :", test_ratio)
        print("Diff    :", diff)
        print("JANGAN gunakan hasil eksperimen ini untuk paper sebelum dataset/mask/split disamakan.")
        print("=" * 80)
    else:
        print("Dataset integrity OK. Test positive-pixel ratio masih konsisten dengan paper utama.")

    return df

integrity_df = taac_dataset_integrity_report()


# %%
# ============================================================
# 6. SAFE METRICS AND FAIR THRESHOLD SELECTION
# ============================================================

def binary_metrics_np(y_true, y_prob, threshold=0.5):
    yt = (np.asarray(y_true).reshape(-1) >= 0.5).astype(np.uint8)

    yp_prob = np.asarray(y_prob).reshape(-1).astype(np.float32)
    yp_prob = np.nan_to_num(yp_prob, nan=0.0, posinf=1.0, neginf=0.0)
    yp_prob = np.clip(yp_prob, 0.0, 1.0)

    if yt.shape[0] != yp_prob.shape[0]:
        raise ValueError(f"Shape mismatch: y_true={yt.shape}, y_prob={yp_prob.shape}")

    yp = (yp_prob >= float(threshold)).astype(np.uint8)

    tp = int(np.count_nonzero((yt == 1) & (yp == 1)))
    fp = int(np.count_nonzero((yt == 0) & (yp == 1)))
    fn = int(np.count_nonzero((yt == 1) & (yp == 0)))
    tn = int(np.count_nonzero((yt == 0) & (yp == 0)))

    tp_f, fp_f, fn_f, tn_f = map(float, [tp, fp, fn, tn])
    eps = 1e-7

    precision = tp_f / (tp_f + fp_f + eps)
    recall = tp_f / (tp_f + fn_f + eps)
    specificity = tn_f / (tn_f + fp_f + eps)
    fpr = fp_f / (fp_f + tn_f + eps)
    dice = (2.0 * tp_f) / (2.0 * tp_f + fp_f + fn_f + eps)
    iou = tp_f / (tp_f + fp_f + fn_f + eps)
    accuracy = (tp_f + tn_f) / (tp_f + tn_f + fp_f + fn_f + eps)
    balanced_accuracy = 0.5 * (recall + specificity)

    denom = (tp_f + fp_f) * (tp_f + fn_f) * (tn_f + fp_f) * (tn_f + fn_f)
    if denom <= 0:
        mcc = 0.0
    else:
        mcc = ((tp_f * tn_f) - (fp_f * fn_f)) / (math.sqrt(denom) + eps)

    return {
        "threshold": float(threshold),
        "iou": float(iou),
        "dice_f1": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "fpr": float(fpr),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "mcc": float(mcc),
        "actual_positive_ratio": float(yt.mean()),
        "predicted_positive_ratio": float(yp.mean()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def pr_auc_score_np(y_true, y_prob):
    yt = (np.asarray(y_true).reshape(-1) >= 0.5).astype(np.uint8)
    yp = np.asarray(y_prob).reshape(-1).astype(np.float32)
    yp = np.nan_to_num(yp, nan=0.0, posinf=1.0, neginf=0.0)
    yp = np.clip(yp, 0.0, 1.0)
    if len(np.unique(yt)) < 2:
        return np.nan
    precision, recall, _ = precision_recall_curve(yt, yp)
    return float(auc(recall, precision))


def select_threshold_constrained_fair(
    y_true,
    y_prob,
    model_name="model",
    threshold_values=THRESHOLD_VALUES,
    lower_mult=RATIO_LOWER_MULT,
    upper_mult=RATIO_UPPER_MULT,
):
    actual_ratio = float((np.asarray(y_true).reshape(-1) >= 0.5).mean())
    rows = []

    for th in threshold_values:
        m = binary_metrics_np(y_true, y_prob, threshold=float(th))
        pred_ratio = m["predicted_positive_ratio"]
        ratio_penalty = abs(np.log((pred_ratio + 1e-7) / (actual_ratio + 1e-7)))

        m["model"] = model_name
        m["positive_ratio_penalty"] = float(ratio_penalty)
        m["is_ratio_valid"] = bool(
            (pred_ratio >= lower_mult * actual_ratio) and
            (pred_ratio <= upper_mult * actual_ratio)
        )

        m["selection_score"] = (
            m["iou"]
            + 0.20 * m["dice_f1"]
            + 0.10 * m["balanced_accuracy"]
            + 0.10 * m["mcc"]
            - 0.025 * ratio_penalty
        )
        rows.append(m)

    df = pd.DataFrame(rows)
    valid = df[df["is_ratio_valid"]].copy()

    if len(valid) > 0:
        best_idx = valid["selection_score"].idxmax()
    else:
        best_idx = df["selection_score"].idxmax()

    best_th = float(df.loc[best_idx, "threshold"])
    return best_th, df


def predict_full(model, X_arr, input_type="sequence", batch_size=BATCH_SIZE, verbose=0):
    if input_type == "2d_last_frame":
        return model.predict(X_arr[:, -1, ...], batch_size=batch_size, verbose=verbose)
    return model.predict(X_arr, batch_size=batch_size, verbose=verbose)


def predict_with_cache(model, X_arr, cache_name, input_type="sequence", force=False):
    cache_path = CACHE_DIR / f"{cache_name}.npy"
    if cache_path.exists() and not force:
        arr = np.load(cache_path)
        print("Loaded prediction cache:", cache_path)
        return arr
    pred = predict_full(model, X_arr, input_type=input_type, batch_size=BATCH_SIZE, verbose=1)
    pred = np.asarray(pred, dtype=np.float32)
    np.save(cache_path, pred)
    print("Saved prediction cache:", cache_path)
    return pred

print("Safe metrics ready.")


# %%
# ============================================================
# 7. LOSS FUNCTIONS, DATASET BUILDER, FALLBACK ST-HRPNET BUILDER
# ============================================================

SMOOTH = 1e-6

if "LOSS_POS_WEIGHT" not in globals():
    source_y = globals().get("y_train_patch", globals().get("y_train", None))
    if source_y is not None:
        pos_ratio = float(np.mean(source_y))
        neg_ratio = 1.0 - pos_ratio
        LOSS_POS_WEIGHT = min(max(neg_ratio / (pos_ratio + 1e-7), 1.0), 20.0)
    else:
        LOSS_POS_WEIGHT = 8.0

print("LOSS_POS_WEIGHT:", LOSS_POS_WEIGHT)


def dice_coef(y_true, y_pred):
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    inter = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * inter + SMOOTH) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + SMOOTH)


def iou_coef(y_true, y_pred):
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    inter = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - inter
    return (inter + SMOOTH) / (union + SMOOTH)


def weighted_bce_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-6, 1.0 - 1e-6)
    bce = keras.backend.binary_crossentropy(y_true, y_pred)
    weights = y_true * LOSS_POS_WEIGHT + (1.0 - y_true)
    return tf.reduce_mean(bce * weights)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def false_positive_penalty(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    bg = 1.0 - y_true
    return tf.reduce_sum(bg * y_pred) / (tf.reduce_sum(bg) + SMOOTH)


def combined_hrp_loss(y_true, y_pred):
    return 0.60 * weighted_bce_loss(y_true, y_pred) + 0.35 * dice_loss(y_true, y_pred) + 0.05 * false_positive_penalty(y_true, y_pred)


def combined_hrp_loss_no_fp(y_true, y_pred):
    return 0.65 * weighted_bce_loss(y_true, y_pred) + 0.35 * dice_loss(y_true, y_pred)


def precision_at_05(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred_bin = tf.cast(y_pred >= 0.5, tf.float32)
    tp = tf.reduce_sum(y_true * y_pred_bin)
    fp = tf.reduce_sum((1.0 - y_true) * y_pred_bin)
    return (tp + SMOOTH) / (tp + fp + SMOOTH)


def recall_at_05(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred_bin = tf.cast(y_pred >= 0.5, tf.float32)
    tp = tf.reduce_sum(y_true * y_pred_bin)
    fn = tf.reduce_sum(y_true * (1.0 - y_pred_bin))
    return (tp + SMOOTH) / (tp + fn + SMOOTH)


def compile_model_with_loss(model, loss_fn="with_fp", lr=1e-4):
    if loss_fn == "no_fp":
        loss = combined_hrp_loss_no_fp
    elif loss_fn == "bce":
        loss = "binary_crossentropy"
    else:
        loss = combined_hrp_loss

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=loss,
        metrics=[dice_coef, iou_coef, precision_at_05, recall_at_05],
    )
    return model


def make_dataset(X_arr, y_arr, batch_size=BATCH_SIZE, shuffle=False):
    X_arr = np.asarray(X_arr, dtype=np.float32)
    y_arr = np.asarray(y_arr, dtype=np.float32)
    ds = tf.data.Dataset.from_tensor_slices((X_arr, y_arr))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(X_arr), 2048), seed=SEED, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def conv2d_block(x, filters, name, dropout=0.0):
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation("relu", name=f"{name}_relu1")(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.Activation("relu", name=f"{name}_relu2")(x)
    if dropout and dropout > 0:
        x = layers.Dropout(dropout, name=f"{name}_drop")(x)
    return x


def td_conv_block(x, filters, name, dropout=0.0):
    x = layers.TimeDistributed(layers.Conv2D(filters, 3, padding="same", use_bias=False), name=f"{name}_td_conv1")(x)
    x = layers.TimeDistributed(layers.BatchNormalization(), name=f"{name}_td_bn1")(x)
    x = layers.TimeDistributed(layers.Activation("relu"), name=f"{name}_td_relu1")(x)
    x = layers.TimeDistributed(layers.Conv2D(filters, 3, padding="same", use_bias=False), name=f"{name}_td_conv2")(x)
    x = layers.TimeDistributed(layers.BatchNormalization(), name=f"{name}_td_bn2")(x)
    x = layers.TimeDistributed(layers.Activation("relu"), name=f"{name}_td_relu2")(x)
    if dropout and dropout > 0:
        x = layers.TimeDistributed(layers.Dropout(dropout), name=f"{name}_td_drop")(x)
    return x


def last_time_step(x):
    return x[:, -1, ...]


def fallback_build_st_hrpnet_v1_patchbalanced(seq_len=SEQ_LEN, n_channels=len(INPUT_PRODUCTS), base=BASE_FILTERS):
    inputs = keras.Input(shape=(seq_len, None, None, n_channels), name="himawari_sequence_dynamic")

    x1 = td_conv_block(inputs, base, "enc1", dropout=0.03)
    p1 = layers.TimeDistributed(layers.MaxPooling2D((2, 2)), name="pool1")(x1)

    x2 = td_conv_block(p1, base * 2, "enc2", dropout=0.05)
    p2 = layers.TimeDistributed(layers.MaxPooling2D((2, 2)), name="pool2")(x2)

    x3 = td_conv_block(p2, base * 4, "enc3", dropout=0.08)
    p3 = layers.TimeDistributed(layers.MaxPooling2D((2, 2)), name="pool3")(x3)

    z = layers.ConvLSTM2D(base * 8, 3, padding="same", return_sequences=False, name="convlstm_bottleneck")(p3)
    z = layers.BatchNormalization(name="bottleneck_bn")(z)
    z = conv2d_block(z, base * 8, "bottleneck_refine", dropout=0.10)

    s3 = layers.Lambda(last_time_step, name="skip3_last")(x3)
    s2 = layers.Lambda(last_time_step, name="skip2_last")(x2)
    s1 = layers.Lambda(last_time_step, name="skip1_last")(x1)

    d3 = layers.UpSampling2D((2, 2), name="up3")(z)
    d3 = layers.Concatenate(name="concat3")([d3, s3])
    d3 = conv2d_block(d3, base * 4, "dec3", dropout=0.08)

    d2 = layers.UpSampling2D((2, 2), name="up2")(d3)
    d2 = layers.Concatenate(name="concat2")([d2, s2])
    d2 = conv2d_block(d2, base * 2, "dec2", dropout=0.05)

    d1 = layers.UpSampling2D((2, 2), name="up1")(d2)
    d1 = layers.Concatenate(name="concat1")([d1, s1])
    d1 = conv2d_block(d1, base, "dec1", dropout=0.03)

    outputs = layers.Conv2D(1, 1, activation="sigmoid", name="hrp_probability")(d1)
    return keras.Model(inputs, outputs, name="ST_HRPNet_v1_PatchBalanced")


if "build_st_hrpnet_v1_patchbalanced" not in globals():
    build_st_hrpnet_v1_patchbalanced = fallback_build_st_hrpnet_v1_patchbalanced
    print("Using fallback ST-HRPNet builder.")
else:
    print("Using existing ST-HRPNet builder from main notebook.")

print("Losses/model helpers ready.")


# %%
# ============================================================
# 8. LOAD ST-HRPNET MODEL / WEIGHTS
# ============================================================

def find_first_existing(paths):
    for p in paths:
        p = Path(p)
        if p.exists():
            return p
    return None


def load_st_model_prefer_memory():
    if "st_model" in globals() and st_model is not None:
        print("Using st_model already in memory.")
        return st_model

    full_model_candidates = [
        CHECKPOINT_DIR / "st_model.keras",
        MAIN_OUTPUT_DIR / "ST_HRPNet_v1_PatchBalanced" / "ST_HRPNet_v1_PatchBalanced.keras",
        MODELS_DIR / "ST_HRPNet_v1_PatchBalanced.keras",
    ]

    full_model_path = find_first_existing(full_model_candidates)
    if full_model_path is not None:
        print("Loading full Keras model:", full_model_path)
        model = keras.models.load_model(
            full_model_path,
            custom_objects={
                "dice_coef": dice_coef,
                "iou_coef": iou_coef,
                "weighted_bce_loss": weighted_bce_loss,
                "dice_loss": dice_loss,
                "false_positive_penalty": false_positive_penalty,
                "combined_hrp_loss": combined_hrp_loss,
                "precision_at_05": precision_at_05,
                "recall_at_05": recall_at_05,
            },
            compile=False,
        )
        globals()["st_model"] = model
        return model

    weights_candidates = [
        MAIN_OUTPUT_DIR / "ST_HRPNet_v1_PatchBalanced" / "ST_HRPNet_v1_PatchBalanced_best.weights.h5",
        CHECKPOINT_DIR / "ST_HRPNet_v1_PatchBalanced_best.weights.h5",
        MODELS_DIR / "ST_HRPNet_v1_PatchBalanced" / "ST_HRPNet_v1_PatchBalanced_best.weights.h5",
    ]

    weights_path = find_first_existing(weights_candidates)
    if weights_path is None:
        raise FileNotFoundError(
            "Tidak menemukan full model atau weights ST-HRPNet. "
            "Pastikan MAIN_OUTPUT_DIR benar atau jalankan training utama dulu."
        )

    print("Loading ST-HRPNet weights:", weights_path)
    model = build_st_hrpnet_v1_patchbalanced()
    try:
        model.load_weights(str(weights_path))
    except Exception as e:
        raise RuntimeError(
            "Gagal load weights. Kemungkinan builder arsitektur tidak sama dengan arsitektur training. "
            "Solusi: jalankan cell arsitektur model dari notebook utama sebelum cell ini, "
            "atau simpan full model .keras dari notebook utama."
        ) from e

    globals()["st_model"] = model
    return model


st_model = load_st_model_prefer_memory()
print("ST model ready:", st_model.name)


# %%
# ============================================================
# 9. MAIN MODEL RE-EVALUATION WITH FAIR THRESHOLD
# ============================================================

def evaluate_model_fair(model, model_name, input_type="sequence", force_predict=False):
    val_prob = predict_with_cache(
        model, X_val, f"{model_name}_validation_prob",
        input_type=input_type, force=force_predict
    )
    test_prob = predict_with_cache(
        model, X_test, f"{model_name}_test_prob",
        input_type=input_type, force=force_predict
    )

    best_th, th_df = select_threshold_constrained_fair(y_val, val_prob, model_name=model_name)
    th_path = TABLES_DIR / f"{model_name}_fair_threshold_search_validation.csv"
    th_df.to_csv(th_path, index=False)

    metrics = binary_metrics_np(y_test, test_prob, threshold=best_th)
    metrics["model"] = model_name
    metrics["selected_threshold_from_validation"] = best_th
    metrics["pr_auc"] = pr_auc_score_np(y_test, test_prob)
    metrics["input_type"] = input_type

    metrics_05 = binary_metrics_np(y_test, test_prob, threshold=0.5)
    metrics_05["model"] = model_name + "_FixedThreshold_0_5"
    metrics_05["selected_threshold_from_validation"] = 0.5
    metrics_05["pr_auc"] = pr_auc_score_np(y_test, test_prob)
    metrics_05["input_type"] = input_type

    return metrics, metrics_05, th_df, val_prob, test_prob


main_metrics, main_metrics_05, main_th_df, st_val_prob, st_test_prob = evaluate_model_fair(
    st_model,
    "Full_STHRPNet_PatchBalanced_FPaware",
    input_type="sequence",
    force_predict=False,
)

main_eval_df = pd.DataFrame([main_metrics_05, main_metrics])
main_eval_path = TABLES_DIR / "main_model_re_evaluation_fair_threshold.csv"
main_eval_df.to_csv(main_eval_path, index=False)

display(main_eval_df)
print("Saved:", main_eval_path)
print("Selected fair threshold:", main_metrics["selected_threshold_from_validation"])


# %%
# ============================================================
# 10. THRESHOLD SENSITIVITY ANALYSIS
# ============================================================

if RUN_THRESHOLD_SENSITIVITY:
    rows = []
    for th in THRESHOLD_VALUES:
        m_val = binary_metrics_np(y_val, st_val_prob, threshold=float(th))
        m_test = binary_metrics_np(y_test, st_test_prob, threshold=float(th))
        rows.append({
            "threshold": float(th),
            "val_iou": m_val["iou"],
            "val_dice_f1": m_val["dice_f1"],
            "val_precision": m_val["precision"],
            "val_recall": m_val["recall"],
            "val_mcc": m_val["mcc"],
            "val_predicted_positive_ratio": m_val["predicted_positive_ratio"],
            "test_iou": m_test["iou"],
            "test_dice_f1": m_test["dice_f1"],
            "test_precision": m_test["precision"],
            "test_recall": m_test["recall"],
            "test_mcc": m_test["mcc"],
            "test_predicted_positive_ratio": m_test["predicted_positive_ratio"],
        })

    threshold_df = pd.DataFrame(rows)
    threshold_path = TABLES_DIR / "threshold_sensitivity_STHRPNet_fair.csv"
    threshold_df.to_csv(threshold_path, index=False)
    display(threshold_df.head())
    print("Saved:", threshold_path)

    plt.figure(figsize=(8, 5))
    plt.plot(threshold_df["threshold"], threshold_df["test_iou"], label="IoU")
    plt.plot(threshold_df["threshold"], threshold_df["test_dice_f1"], label="Dice/F1")
    plt.plot(threshold_df["threshold"], threshold_df["test_mcc"], label="MCC")
    plt.axvline(main_metrics["selected_threshold_from_validation"], linestyle="--", label="Selected threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Metric value")
    plt.title("Threshold sensitivity of ST-HRPNet v1")
    plt.legend()
    plt.grid(True, alpha=0.3)
    fig_path = FIGURES_DIR / "threshold_sensitivity_metrics.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.show()
    print("Saved:", fig_path)

    actual_ratio_test = float(np.mean(y_test))
    plt.figure(figsize=(8, 5))
    plt.plot(threshold_df["threshold"], threshold_df["test_predicted_positive_ratio"], label="Predicted positive ratio")
    plt.axhline(actual_ratio_test, linestyle="--", label="Actual test positive ratio")
    plt.axvline(main_metrics["selected_threshold_from_validation"], linestyle="--", label="Selected threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Positive-pixel ratio")
    plt.title("Positive-ratio behavior under threshold variation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    fig_path = FIGURES_DIR / "threshold_sensitivity_positive_ratio.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.show()
    print("Saved:", fig_path)
else:
    print("Threshold sensitivity skipped.")


# %%
# ============================================================
# 11. FAIR ABLATION HELPERS
# ============================================================

def make_random_patches(X_arr, y_arr, n_patches, patch_size=PATCH_SIZE):
    Xp, yp = [], []
    n, t, h, w, c = X_arr.shape
    for _ in tqdm(range(int(n_patches)), desc="Random patches"):
        idx = np.random.randint(0, n)
        top = np.random.randint(0, h - patch_size + 1)
        left = np.random.randint(0, w - patch_size + 1)
        Xp.append(X_arr[idx, :, top:top + patch_size, left:left + patch_size, :])
        yp.append(y_arr[idx, top:top + patch_size, left:left + patch_size, :])
    return np.asarray(Xp, dtype=np.float32), np.asarray(yp, dtype=np.float32)


def ensure_patch_data():
    global X_train_patch, y_train_patch, X_val_patch, y_val_patch

    if all(v in globals() for v in ["X_train_patch", "y_train_patch", "X_val_patch", "y_val_patch"]):
        print("Patch-balanced data already in memory.")
        return

    files = {name: CHECKPOINT_DIR / f"{name}.npy" for name in ["X_train_patch", "y_train_patch", "X_val_patch", "y_val_patch"]}
    if all(p.exists() for p in files.values()):
        X_train_patch = np.load(files["X_train_patch"])
        y_train_patch = np.load(files["y_train_patch"])
        X_val_patch = np.load(files["X_val_patch"])
        y_val_patch = np.load(files["y_val_patch"])
        print("Patch-balanced checkpoints loaded.")
        return

    print("Patch-balanced arrays belum ada. Membuat fallback random patch.")
    print("Untuk paper, lebih baik gunakan patch-balanced arrays asli dari notebook utama.")
    n_train = globals().get("N_TRAIN_PATCHES", 1800)
    n_val = globals().get("N_VAL_PATCHES", 450)
    X_train_patch, y_train_patch = make_random_patches(X_train, y_train, n_train, PATCH_SIZE)
    X_val_patch, y_val_patch = make_random_patches(X_val, y_val, n_val, PATCH_SIZE)

    for name in ["X_train_patch", "y_train_patch", "X_val_patch", "y_val_patch"]:
        np.save(CHECKPOINT_DIR / f"{name}.npy", globals()[name])


def train_model_safe(model, train_data, val_data, name, epochs=EPOCHS_ABLATION):
    model_dir = MODELS_DIR / name
    model_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = model_dir / f"{name}_best.weights.h5"
    history_path = TABLES_DIR / f"{name}_history.csv"
    log_path = TABLES_DIR / f"{name}_training_log.csv"

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_iou_coef",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_iou_coef",
            mode="max",
            patience=ABLATION_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(str(log_path), append=False),
    ]

    start = time.time()
    hist = model.fit(
        train_data,
        validation_data=val_data,
        epochs=int(epochs),
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.time() - start

    hist_df = pd.DataFrame(hist.history)
    hist_df["epoch"] = np.arange(1, len(hist_df) + 1)
    hist_df["training_time_seconds_total"] = elapsed
    hist_df.to_csv(history_path, index=False)

    if ckpt_path.exists():
        model.load_weights(str(ckpt_path))

    print("Training done:", name)
    print("Weights:", ckpt_path)
    return model, hist, model_dir


# %%
# ============================================================
# 12. FAIR ABLATION STUDY
# ============================================================

def build_or_load_ablation_model(variant_name, loss_fn, train_ds=None, val_ds=None):
    model_dir = MODELS_DIR / variant_name
    weights_path = model_dir / f"{variant_name}_best.weights.h5"

    model = build_st_hrpnet_v1_patchbalanced()
    model = compile_model_with_loss(model, loss_fn=loss_fn, lr=1e-4)

    if weights_path.exists() and not RETRAIN_ABLATIONS:
        print("Loading existing ablation weights:", weights_path)
        model.load_weights(str(weights_path))
        return model

    if not TRAIN_MISSING_ABLATIONS:
        raise FileNotFoundError(f"Weights ablation belum ada dan TRAIN_MISSING_ABLATIONS=False: {weights_path}")

    if train_ds is None or val_ds is None:
        raise RuntimeError("train_ds dan val_ds wajib tersedia untuk training ablation.")

    model, hist, model_dir = train_model_safe(model, train_ds, val_ds, variant_name, epochs=EPOCHS_ABLATION)
    return model


def evaluate_ablation_variant(model, model_name, input_type="sequence"):
    metrics, metrics_05, th_df, val_prob, test_prob = evaluate_model_fair(
        model, model_name, input_type=input_type, force_predict=False
    )
    metrics["ablation_setting"] = model_name
    metrics_05["ablation_setting"] = model_name + "_fixed_0_5"
    return metrics, metrics_05


if RUN_FAIR_ABLATION:
    ensure_patch_data()

    n_train_patches = len(X_train_patch)
    n_val_patches = len(X_val_patch)

    X_train_random_patch, y_train_random_patch = make_random_patches(X_train, y_train, n_train_patches, PATCH_SIZE)
    X_val_random_patch, y_val_random_patch = make_random_patches(X_val, y_val, n_val_patches, PATCH_SIZE)

    train_random_ds = make_dataset(X_train_random_patch, y_train_random_patch, shuffle=True)
    val_random_ds = make_dataset(X_val_random_patch, y_val_random_patch, shuffle=False)

    train_balanced_ds = make_dataset(X_train_patch, y_train_patch, shuffle=True)
    val_balanced_ds = make_dataset(X_val_patch, y_val_patch, shuffle=False)

    ablation_rows = [main_metrics_05, main_metrics]

    try:
        model_A = build_or_load_ablation_model("A_RandomPatch_WBCE_Dice_NoFP", "no_fp", train_random_ds, val_random_ds)
        mA, mA05 = evaluate_ablation_variant(model_A, "A_RandomPatch_WBCE_Dice_NoFP")
        ablation_rows.extend([mA05, mA])
    except Exception as e:
        print("Ablation A skipped/failed:", e)

    try:
        model_B = build_or_load_ablation_model("B_PatchBalanced_WBCE_Dice_NoFP", "no_fp", train_balanced_ds, val_balanced_ds)
        mB, mB05 = evaluate_ablation_variant(model_B, "B_PatchBalanced_WBCE_Dice_NoFP")
        ablation_rows.extend([mB05, mB])
    except Exception as e:
        print("Ablation B skipped/failed:", e)

    try:
        model_C = build_or_load_ablation_model("C_PatchBalanced_FPaware", "with_fp", train_balanced_ds, val_balanced_ds)
        mC, mC05 = evaluate_ablation_variant(model_C, "C_PatchBalanced_FPaware")
        ablation_rows.extend([mC05, mC])
    except Exception as e:
        print("Ablation C skipped/failed:", e)

    ablation_df = pd.DataFrame(ablation_rows)

    preferred_cols = [
        "model", "ablation_setting", "input_type", "selected_threshold_from_validation",
        "threshold", "iou", "dice_f1", "precision", "recall", "specificity", "fpr",
        "accuracy", "balanced_accuracy", "mcc", "pr_auc",
        "actual_positive_ratio", "predicted_positive_ratio", "tp", "fp", "fn", "tn"
    ]
    existing_cols = [c for c in preferred_cols if c in ablation_df.columns]
    other_cols = [c for c in ablation_df.columns if c not in existing_cols]
    ablation_df = ablation_df[existing_cols + other_cols]

    ablation_path = TABLES_DIR / "fair_ablation_study_results.csv"
    ablation_df.to_csv(ablation_path, index=False)
    display(ablation_df)
    print("Saved:", ablation_path)

    plot_df = ablation_df[~ablation_df["model"].astype(str).str.contains("FixedThreshold", na=False)].copy()
    if not plot_df.empty:
        plt.figure(figsize=(10, 5))
        x = np.arange(len(plot_df))
        plt.bar(x, plot_df["iou"])
        plt.xticks(x, plot_df["model"], rotation=35, ha="right")
        plt.ylabel("IoU")
        plt.title("Fair ablation using validation-selected constrained threshold")
        plt.tight_layout()
        fig_path = FIGURES_DIR / "fair_ablation_iou.png"
        plt.savefig(fig_path, dpi=300)
        plt.show()
        print("Saved:", fig_path)
else:
    print("Fair ablation skipped.")


# %%
# ============================================================
# 13. OPTIONAL INDEPENDENT PRECIPITATION VALIDATION
# ============================================================

def load_external_precip_mask():
    npy_path = EXTERNAL_PRECIP_DIR / "precip_test_masks_256.npy"
    if not npy_path.exists():
        status = {
            "status": "not_available",
            "message": "No external precipitation mask found. Put precip_test_masks_256.npy in external_precip_validation.",
            "path_checked": str(npy_path),
        }
        with open(TABLES_DIR / "external_precip_validation_status.json", "w", encoding="utf-8") as f:
            json.dump(status, f, indent=2)
        print(status["message"])
        return None

    arr = np.load(npy_path)
    arr = (arr >= 0.5).astype(np.uint8)
    if arr.ndim == 3:
        arr = arr[..., None]
    if arr.shape != y_test.shape:
        raise ValueError(f"External precip mask shape {arr.shape} tidak sama dengan y_test {y_test.shape}.")
    return arr


if RUN_EXTERNAL_PRECIP_VALIDATION:
    precip_mask = load_external_precip_mask()

    if precip_mask is not None:
        selected_th = float(main_metrics["selected_threshold_from_validation"])
        pred_mask = (st_test_prob >= selected_th).astype(np.uint8)

        ext_metrics = binary_metrics_np(precip_mask, pred_mask, threshold=0.5)
        ext_metrics["model"] = "Full_STHRPNet_vs_ExternalPrecipMask"
        ext_metrics["reference"] = "external_precipitation_binary_mask"
        ext_metrics["selected_threshold_from_validation"] = selected_th

        ext_df = pd.DataFrame([ext_metrics])
        ext_path = TABLES_DIR / "external_precip_validation_metrics.csv"
        ext_df.to_csv(ext_path, index=False)
        display(ext_df)
        print("Saved:", ext_path)
else:
    print("External precipitation validation skipped.")


# %%
# ============================================================
# 14. EVENT-BASED METEOROLOGICAL CASE ANALYSIS
# ============================================================

def select_event_indices(k=5):
    ratios = np.asarray([float(y_test[i].mean()) for i in range(len(y_test))])
    order = np.argsort(-ratios)
    return order[:min(k, len(order))]


def get_time_label(idx):
    if "meta_test" in globals() and isinstance(meta_test, pd.DataFrame) and idx < len(meta_test):
        row = meta_test.iloc[idx]
        for col in ["target_time_code", "time_code", "end_time_code", "time"]:
            if col in row:
                return str(row[col])
    return f"test_index_{idx}"


def visualize_event_case(idx, threshold=None):
    if threshold is None:
        threshold = float(main_metrics["selected_threshold_from_validation"])

    x_last = X_test[idx, -1]
    y_true = y_test[idx, ..., 0] if y_test.ndim == 4 else y_test[idx]
    prob = st_test_prob[idx, ..., 0] if st_test_prob.ndim == 4 else st_test_prob[idx]
    pred = (prob >= threshold).astype(np.uint8)

    b13 = x_last[..., 0]

    error = np.zeros_like(y_true, dtype=np.float32)
    error[(y_true >= 0.5) & (pred == 1)] = 1.0
    error[(y_true < 0.5) & (pred == 1)] = 0.6
    error[(y_true >= 0.5) & (pred == 0)] = 0.3

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))

    axes[0].imshow(b13, cmap="gray")
    axes[0].set_title("Last B13/input")
    axes[0].axis("off")

    axes[1].imshow(y_true, cmap="gray")
    axes[1].set_title("HRP target")
    axes[1].axis("off")

    axes[2].imshow(prob, cmap="viridis")
    axes[2].set_title("Probability")
    axes[2].axis("off")

    axes[3].imshow(pred, cmap="gray")
    axes[3].set_title(f"Prediction th={threshold:.2f}")
    axes[3].axis("off")

    axes[4].imshow(error, cmap="magma")
    axes[4].set_title("Error map")
    axes[4].axis("off")

    time_label = get_time_label(idx)
    fig.suptitle(f"Event case: {time_label} | test index={idx} | target ratio={float(y_true.mean()):.4f}")

    safe_label = re.sub(r"[^A-Za-z0-9_\\-]+", "_", str(time_label))
    out_path = FIGURES_DIR / f"event_case_{idx:03d}_{safe_label}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.show()

    return out_path


if RUN_EVENT_CASE_ANALYSIS:
    selected_indices = select_event_indices(k=5)
    event_rows = []

    for idx in selected_indices:
        out_path = visualize_event_case(int(idx))
        event_rows.append({
            "test_index": int(idx),
            "time_label": get_time_label(int(idx)),
            "target_positive_ratio": float(np.mean(y_test[int(idx)])),
            "figure_path": str(out_path),
        })

    event_df = pd.DataFrame(event_rows)
    event_path = TABLES_DIR / "event_case_selection.csv"
    event_df.to_csv(event_path, index=False)
    display(event_df)
    print("Saved:", event_path)
else:
    print("Event case analysis skipped.")


# %%
# ============================================================
# 15. EXPORT MANUSCRIPT-READY SUMMARY
# ============================================================

def read_csv_safe(path):
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


summary_lines = []
summary_lines.append("# TAAC Re-Experiment V3 Summary")
summary_lines.append("")
summary_lines.append(f"Paper title: {PAPER_TITLE_TAAC}")
summary_lines.append("")

summary_lines.append("## Dataset integrity")
summary_lines.append(integrity_df.to_markdown(index=False))

main_eval_df = read_csv_safe(TABLES_DIR / "main_model_re_evaluation_fair_threshold.csv")
if not main_eval_df.empty:
    summary_lines.append("")
    summary_lines.append("## Main model re-evaluation")
    summary_lines.append(main_eval_df.to_markdown(index=False))

ablation_df = read_csv_safe(TABLES_DIR / "fair_ablation_study_results.csv")
if not ablation_df.empty:
    summary_lines.append("")
    summary_lines.append("## Fair ablation study")
    summary_lines.append(ablation_df.to_markdown(index=False))

threshold_df = read_csv_safe(TABLES_DIR / "threshold_sensitivity_STHRPNet_fair.csv")
if not threshold_df.empty:
    best_row = threshold_df.sort_values("test_iou", ascending=False).head(1)
    summary_lines.append("")
    summary_lines.append("## Threshold sensitivity")
    summary_lines.append("Best test-IoU threshold row:")
    summary_lines.append(best_row.to_markdown(index=False))

ext_df = read_csv_safe(TABLES_DIR / "external_precip_validation_metrics.csv")
if not ext_df.empty:
    summary_lines.append("")
    summary_lines.append("## External precipitation validation")
    summary_lines.append(ext_df.to_markdown(index=False))
else:
    summary_lines.append("")
    summary_lines.append("## External precipitation validation")
    summary_lines.append("No valid external precipitation mask was available. Do not claim independent precipitation validation.")

event_df = read_csv_safe(TABLES_DIR / "event_case_selection.csv")
if not event_df.empty:
    summary_lines.append("")
    summary_lines.append("## Event case figures")
    summary_lines.append(event_df.to_markdown(index=False))

summary_path = TAAC_OUTPUT_DIR / "TAAC_reexperiment_v3_summary.md"
summary_path.write_text("\\n".join(summary_lines), encoding="utf-8")

print("Summary saved:", summary_path)
print("\\n".join(summary_lines[:40]))


# %%
# ============================================================
# 16. FINAL DECISION CHECK
# ============================================================

def final_decision_check():
    issues = []

    test_ratio = float(np.mean(y_test))
    if abs(test_ratio - EXPECTED_TEST_POS_RATIO) > POS_RATIO_TOLERANCE:
        issues.append(
            f"Test positive ratio mismatch: current={test_ratio:.6f}, expected~={EXPECTED_TEST_POS_RATIO:.6f}"
        )

    main_df = read_csv_safe(TABLES_DIR / "main_model_re_evaluation_fair_threshold.csv")
    if main_df.empty:
        issues.append("Main re-evaluation table missing.")
    else:
        row = main_df[~main_df["model"].astype(str).str.contains("FixedThreshold", na=False)].tail(1)
        if len(row) > 0:
            iou = float(row.iloc[0]["iou"])
            mcc = float(row.iloc[0]["mcc"])
            pr = float(row.iloc[0]["predicted_positive_ratio"])
            ar = float(row.iloc[0]["actual_positive_ratio"])
            if iou <= 0:
                issues.append("Main IoU is non-positive.")
            if mcc <= 0:
                issues.append("Main MCC is non-positive.")
            if not (0.5 * ar <= pr <= 2.0 * ar):
                issues.append(f"Predicted positive ratio not within 0.5x-2x actual: pred={pr:.6f}, actual={ar:.6f}")

    ab_df = read_csv_safe(TABLES_DIR / "fair_ablation_study_results.csv")
    if ab_df.empty:
        issues.append("Fair ablation table missing. Bisa ditunda, tetapi novelty model lebih lemah tanpa ablation.")

    ext_df = read_csv_safe(TABLES_DIR / "external_precip_validation_metrics.csv")
    if ext_df.empty:
        print("External precipitation validation not available. Treat it as limitation/future work, not as result.")

    if issues:
        print("=" * 80)
        print("BELUM AMAN UNTUK NARASI HASIL FINAL.")
        print("=" * 80)
        for issue in issues:
            print("-", issue)
    else:
        print("=" * 80)
        print("AMAN UNTUK LANJUT KE PENULISAN RESULTS/DISCUSSION.")
        print("=" * 80)
        print("Gunakan tabel dan figure dari:", TAAC_OUTPUT_DIR)

final_decision_check()

