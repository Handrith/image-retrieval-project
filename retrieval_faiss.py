import os
import time
import faiss
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

# ======================================
# KONFIGURASI
# ======================================
TEST_DIR = "dataset/test"
INDEX_DIR = "faiss_index"
TOP_K = 50
IMG_SIZE = (224, 224)

MODELS = {
    "resnet": {
        "dim": 2048,
        "weights": "imagenet",
        "model": ResNet50(weights='imagenet', include_top=False, pooling='avg'),
        "preprocess": resnet_preprocess
    },
    "mobilenet": {
        "dim": 1280,
        "weights": "imagenet",
        "model": MobileNetV2(weights='imagenet', include_top=False, pooling='avg'),
        "preprocess": mobilenet_preprocess
    }
}

# ======================================
# FUNGSI PEMBANTU
# ======================================
def load_faiss_index(model_name):
    index_path = os.path.join(INDEX_DIR, f"{model_name}_faiss_orig.index")
    label_path = os.path.join(INDEX_DIR, f"{model_name}_faiss_orig_labels.npy")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index untuk {model_name} tidak ditemukan! Jalankan faiss_index.py dulu.")
    
    index = faiss.read_index(index_path)
    labels = np.load(label_path, allow_pickle=True)
    return index, labels

def recall_at_k(query_label, retrieved_labels, all_labels, k):
    """
    Hitung Recall@K: proporsi item relevan yang berhasil ditemukan dalam top-K
    """
    relevant_total = sum(1 for lbl in all_labels if lbl == query_label)
    if relevant_total == 0:
        return 0.0
    relevant_found = sum(1 for lbl in retrieved_labels[:k] if lbl == query_label)
    return relevant_found / relevant_total


import os

def clean_label_raw(x):
    """Normalize label: basename, remove _orig, remove extension, lowercase, strip."""
    b = os.path.basename(str(x))
    b = os.path.splitext(b)[0]
    b = b.replace("_orig", "")
    return b.strip().lower()

def compute_recall_curve(query_label,
                         retrieved_labels,
                         all_labels,
                         max_k=10,
                         match_mode="prefix",
                         prefix_len=7,
                         fuzzy_threshold=None):
    """
    Hitung recall@k untuk k=1..max_k dengan proteksi index out-of-range.

    Params:
      - query_label: raw filename (will be cleaned by clean_label_raw)
      - retrieved_labels: list of raw labels (will be cleaned)
      - all_labels: list of all labels in gallery (raw or cleaned)
      - max_k: maximum K to compute (will stop at available retrieved length if shorter)
      - match_mode: "prefix" or "exact" or "fuzzy"
      - prefix_len: panjang prefix yang dipakai bila match_mode="prefix"
      - fuzzy_threshold: if not None, enable difflib matching with this threshold (0..1)

    Returns:
      - list of recall values length = max_k (or length limited to max_k but safe)
    """
    # normalize labels
    q = clean_label_raw(query_label)
    retrieved = [clean_label_raw(x) for x in retrieved_labels]
    gallery = [clean_label_raw(x) for x in all_labels]

    # total relevant in gallery (based on chosen basic rule: prefix match)
    if match_mode == "exact":
        total_relevant = sum(1 for lbl in gallery if lbl == q)
    else:
        # default: prefix match
        total_relevant = sum(1 for lbl in gallery if lbl.startswith(q[:prefix_len]))

    if total_relevant == 0:
        # nothing relevant in gallery -> recall is 0 for all k
        return [0.0] * max_k

    recalls = []
    found_relevant = 0
    # iterate k = 1..max_k but guard when retrieved shorter
    for k in range(1, max_k + 1):
        idx = k - 1
        if idx < len(retrieved):
            candidate = retrieved[idx]
            is_relevant = False

            if match_mode == "exact":
                is_relevant = (candidate == q)
            elif match_mode == "prefix":
                is_relevant = candidate.startswith(q[:prefix_len])
            elif match_mode == "fuzzy":
                # fuzzy fallback using difflib if threshold provided
                import difflib
                sim = difflib.SequenceMatcher(None, q, candidate).ratio()
                is_relevant = (sim >= (fuzzy_threshold or 0.75))
            else:
                # fallback to prefix
                is_relevant = candidate.startswith(q[:prefix_len])

            if is_relevant:
                found_relevant += 1
        # else: no retrieved item at this rank -> found_relevant unchanged

        recall = found_relevant / total_relevant
        recalls.append(recall)

    return recalls



def extract_feature(model, preprocess_func, img_path):
    """Ekstraksi fitur untuk satu gambar query"""
    img = image.load_img(img_path, target_size=IMG_SIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_func(x)
    feat = model.predict(x, verbose=0).flatten()
    faiss.normalize_L2(feat.reshape(1, -1))  # normalisasi
    return feat


def search_faiss(model_name, model, preprocess_func):
    print(f"\n🔍 Evaluasi model: {model_name.upper()} dengan FAISS")

    index, labels = load_faiss_index(model_name)

    query_files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not query_files:
        print("❌ Tidak ada file di folder test.")
        return

    for query_name in query_files:
        query_path = os.path.join(TEST_DIR, query_name)
        print(f"\n📸 Query: {query_name}")

        start_time = time.time()
        feat = extract_feature(model, preprocess_func, query_path).reshape(1, -1)
        D, I = index.search(feat, TOP_K)
        end_time = time.time()

        print(f"⚡ Waktu pencarian: {end_time - start_time:.4f} detik")
        for rank, (idx, sim) in enumerate(zip(I[0], D[0]), 1):
            print(f"{rank}. {labels[idx]} | Similarity = {sim:.4f}")


# ======================================
# MAIN PROGRAM
# ======================================
if __name__ == "__main__":
    print("🚀 Retrieval Berbasis FAISS Dimulai...\n")
    for model_name, cfg in MODELS.items():
        search_faiss(model_name, cfg["model"], cfg["preprocess"])
    print("\n✅ Semua proses retrieval selesai.")
