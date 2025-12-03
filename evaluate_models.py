import os
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics.pairwise import cosine_similarity
from retrieval_faiss import extract_feature, load_faiss_index, MODELS
from tqdm import tqdm

# ==========================
# KONFIGURASI
# ==========================
TEST_DIR = r"C:\Imret\dataset\test"
TOP_K = 5

def compute_precision_recall(query_label, retrieved_labels, all_labels, k=5):
    """Hitung precision & recall sederhana."""
    query_prefix = query_label[:7].lower()
    relevant = [lbl for lbl in all_labels if lbl[:7].lower() == query_prefix]

    retrieved = retrieved_labels[:k]
    true_pos = sum(1 for lbl in retrieved if lbl[:7].lower() == query_prefix)
    precision = true_pos / k
    recall = true_pos / len(relevant) if relevant else 0
    return precision, recall

# ==========================
# EVALUASI
# ==========================
def evaluate_model(model_name, model, preprocess, index, labels):
    test_images = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    all_labels = [os.path.splitext(os.path.basename(l))[0].replace("_orig", "") for l in labels]

    total_precision, total_recall, total_time = [], [], []

    for img_name in tqdm(test_images, desc=f"Evaluasi {model_name.upper()}"):
        query_path = os.path.join(TEST_DIR, img_name)
        query_label = os.path.splitext(img_name)[0]

        feat = extract_feature(model, preprocess, query_path).reshape(1, -1)
        start = time.time()
        D, I = index.search(feat, TOP_K)
        elapsed = time.time() - start
        total_time.append(elapsed)

        retrieved_labels = [os.path.splitext(os.path.basename(labels[i]))[0].replace("_orig", "") for i in I[0]]

        prec, rec = compute_precision_recall(query_label, retrieved_labels, all_labels, k=TOP_K)
        total_precision.append(prec)
        total_recall.append(rec)

    return {
        "precision": np.mean(total_precision),
        "recall": np.mean(total_recall),
        "time": np.mean(total_time)
    }

# ==========================
# MAIN
# ==========================
if __name__ == "__main__":
    results = {}

    for model_name, cfg in MODELS.items():
        model = cfg["model"]
        preprocess = cfg["preprocess"]
        index, labels = load_faiss_index(model_name)

        res = evaluate_model(model_name, model, preprocess, index, labels)
        results[model_name] = res

    # ==========================
    # CETAK TABEL HASIL
    # ==========================
    print("\n📊 HASIL PERBANDINGAN MODEL")
    print("="*60)
    print(f"{'Model':<15}{'Precision@5':<15}{'Recall@5':<15}{'Time (s)':<15}")
    print("-"*60)
    for m, r in results.items():
        print(f"{m:<15}{r['precision']:<15.4f}{r['recall']:<15.4f}{r['time']:<15.4f}")

    # ==========================
    # GRAFIK PERBANDINGAN
    # ==========================
    models = list(results.keys())
    precisions = [results[m]["precision"] for m in models]
    recalls = [results[m]["recall"] for m in models]
    times = [results[m]["time"] for m in models]

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].bar(models, precisions, color='skyblue')
    ax[0].set_title("Precision@5")
    ax[0].set_ylim(0, 1)

    ax[1].bar(models, recalls, color='lightgreen')
    ax[1].set_title("Recall@5")
    ax[1].set_ylim(0, 1)

    ax[2].bar(models, times, color='salmon')
    ax[2].set_title("Rata-rata Waktu Retrieval (detik)")

    plt.suptitle("Perbandingan Kinerja Model (ResNet50 vs MobileNetV2)")
    plt.tight_layout()
    plt.savefig("static/evaluation_summary.png")
    plt.show()
