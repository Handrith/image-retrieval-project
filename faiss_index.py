import os
import faiss
import numpy as np

# ================================
# KONFIGURASI
# ================================
FEATURES_DIR = "features"
INDEX_DIR = "faiss_index"
MODELS = ["resnet", "mobilenet"]
AUGMENTED = True   # True kalau pakai hasil augmentasi (tetapi hanya nama file _orig yang diindeks)
DIM = {"resnet": 2048, "mobilenet": 1280}

# ================================
# FUNGSI BANTUAN
# ================================
def build_faiss_index(model_name, use_augmented=True):
    if use_augmented:
        feat_path = os.path.join(FEATURES_DIR, f"{model_name}_aug_features.npy")
        label_path = os.path.join(FEATURES_DIR, f"{model_name}_aug_labels.npy")
    else:
        feat_path = os.path.join(FEATURES_DIR, f"{model_name}_features.npy")
        label_path = os.path.join(FEATURES_DIR, f"{model_name}_labels.npy")

    print(f"\n📦 Memuat fitur {model_name.upper()} dari {feat_path}")
    features = np.load(feat_path, allow_pickle=True)
    labels = np.load(label_path, allow_pickle=True)

    # ✅ Filter hanya label dengan '_orig'
    mask = np.array(["_orig" in lbl for lbl in labels])
    features = features[mask]
    labels = labels[mask]
    print(f"✅ Total fitur setelah filter: {len(features)} (hanya gambar original)")

    # Normalisasi fitur (disarankan untuk cosine similarity)
    faiss.normalize_L2(features)

    # ================================
    # Bangun Index FAISS
    # ================================
    d = DIM[model_name]
    index = faiss.IndexFlatIP(d)  # inner product = cosine similarity (karena sudah dinormalisasi)
    index.add(features)

    # Simpan index dan label
    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, os.path.join(INDEX_DIR, f"{model_name}_faiss_orig.index"))
    np.save(os.path.join(INDEX_DIR, f"{model_name}_faiss_orig_labels.npy"), labels)

    print(f"🎯 Index FAISS untuk {model_name.upper()} selesai dibuat dan disimpan!\n")


# ================================
# MAIN PROGRAM
# ================================
if __name__ == "__main__":
    for model in MODELS:
        build_faiss_index(model, use_augmented=AUGMENTED)
    print("\n🔥 Semua index FAISS berhasil dibuat hanya untuk gambar _orig!")
