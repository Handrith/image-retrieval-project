from flask import Flask, render_template, request, url_for
import os
from retrieval_faiss import extract_feature, load_faiss_index, MODELS, TOP_K, compute_recall_curve
import time
import difflib
import matplotlib
matplotlib.use('Agg')  # non-GUI backend untuk server Flask
import matplotlib.pyplot as plt
import uuid
import numpy as np
import shutil
import glob
import seaborn as sns
from sklearn.metrics import confusion_matrix, average_precision_score
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)

# Folder dataset
TEST_DIR = r"C:\Imret\dataset\test"
TRAIN_DIR = r"C:\Imret\dataset\train"

# ============================================================
# 🧠 Fungsi bantu umum
# ============================================================
def extract_label(filename):
    base = os.path.splitext(os.path.basename(filename))[0]
    if "_orig" in base:
        base = base.replace("_orig", "")
    return base


def find_image_path(label):
    """Cari gambar dengan nama mirip label di folder TRAIN_DIR"""
    pattern = os.path.join(TRAIN_DIR, f"{label}*")
    matches = glob.glob(pattern)
    for m in matches:
        ext = os.path.splitext(m)[1].lower()
        if ext in [".jpg", ".jpeg", ".png"]:
            return m
    return None


def precision_at_k(query_label, retrieved_labels, k=5, threshold=0.7):
    """Hitung precision@k sederhana berbasis similarity nama"""
    query_label = extract_label(query_label).lower()
    relevant = 0
    for retrieved in retrieved_labels[:k]:
        r = extract_label(retrieved).lower()
        sim = difflib.SequenceMatcher(None, query_label, r).ratio()
        if sim >= threshold:
            relevant += 1
    return relevant / k


# ============================================================
# 🔍 FUNGSI RETRIEVAL UTAMA
# ============================================================
def retrieve_results(query_path):
    from utils import generate_scatter_plot  # 🔹 pastikan fungsi ini ada di utils.py
    all_results = {}
    recall_curves = {}

    INDEX_DIR = "faiss_index"
    all_labels = []
    for f in os.listdir(INDEX_DIR):
        if f.endswith("_labels.npy"):
            labels = np.load(os.path.join(INDEX_DIR, f), allow_pickle=True)
            clean_labels = [os.path.splitext(os.path.basename(l))[0].replace("_orig", "") for l in labels]
            all_labels.extend(clean_labels)

    static_results_dir = os.path.join(app.root_path, "static", "results")
    os.makedirs(static_results_dir, exist_ok=True)

    query_label = os.path.splitext(os.path.basename(query_path))[0].replace("_orig", "")

    for model_name, cfg in MODELS.items():
        print(f"\n[INFO] Proses retrieval untuk model: {model_name}")
        model = cfg["model"]
        preprocess = cfg["preprocess"]

        index, labels = load_faiss_index(model_name)
        start_time = time.time()
        feat = extract_feature(model, preprocess, query_path).reshape(1, -1)
        D, I = index.search(feat, TOP_K)
        end_time = time.time()

        results = []
        for rank, (idx, sim) in enumerate(zip(I[0], D[0]), 1):
            clean_label = os.path.splitext(os.path.basename(labels[idx]))[0].replace("_orig", "")
            img_path = find_image_path(clean_label)

            if img_path and os.path.exists(img_path):
                dst_name = f"{uuid.uuid4().hex}_{os.path.basename(img_path)}"
                dst_path = os.path.join(static_results_dir, dst_name)
                shutil.copy(img_path, dst_path)
                img_url = url_for("static", filename=f"results/{dst_name}")
            else:
                print(f"[WARNING] Gambar tidak ditemukan untuk label: {clean_label}")
                img_url = None

            results.append({
                "rank": rank,
                "label": clean_label,
                "similarity": sim,
                "image": img_url
            })

        correct = sum(1 for r in results if r["label"][:7] == query_label[:7])
        precision = (correct / TOP_K) * 100
        retrieved_labels = [r["label"] for r in results]

        recall_curve = compute_recall_curve(query_label, retrieved_labels, all_labels, max_k=10)
        recall_curves[model_name] = recall_curve

        # --- Hitung mAP
        map_score = compute_map(query_label, retrieved_labels, all_labels, k=TOP_K)

        # --- Scatter Plot
        print(f"[DEBUG] Membuat scatter plot untuk {model_name}")
        scatter_img = generate_scatter_plot(
            np.vstack([feat, *index.reconstruct_n(0, len(labels))[:100]]),
            [query_label] + [os.path.splitext(os.path.basename(l))[0] for l in labels[:100]],
            query_label,
            model_name
        )

        # --- Confusion Matrix
        all_true = [query_label for _ in retrieved_labels]
        all_pred = retrieved_labels
        cm_img = generate_confusion_matrix(all_true, all_pred, model_name)

        all_results[model_name] = {
            "time": end_time - start_time,
            "results": results,
            "precision": precision,
            "map": map_score * 100,
            "query_name": os.path.basename(query_path),
            "scatter_plot": scatter_img,
            "confusion_matrix": cm_img,
        }

    # --- Plot Recall Curve ---
    fig, ax = plt.subplots(figsize=(6, 4))
    for model_name, recalls in recall_curves.items():
        ax.plot(range(1, len(recalls) + 1), recalls, marker='o', label=model_name.upper())
    ax.set_xlabel('K')
    ax.set_ylabel('Recall@K')
    ax.set_title('Recall Curve Comparison')
    ax.legend()
    plt.tight_layout()

    static_dir = os.path.join(app.root_path, 'static')
    os.makedirs(static_dir, exist_ok=True)
    graph_path = os.path.join(static_dir, f"recall_curve_{uuid.uuid4().hex}.png")
    plt.savefig(graph_path)
    plt.close()

    all_results["recall_graph"] = os.path.basename(graph_path)
    return all_results


# ============================================================
# 📊 Fungsi Grafik Pendukung
# ============================================================
def generate_confusion_matrix(all_true, all_pred, model_name):
    """Generate confusion matrix sederhana antar prefix label"""
    true_prefixes = [t[:7].upper() for t in all_true]
    pred_prefixes = [p[:7].upper() for p in all_pred]

    unique_labels = sorted(list(set(true_prefixes + pred_prefixes)))
    cm = confusion_matrix(true_prefixes, pred_prefixes, labels=unique_labels)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=unique_labels, yticklabels=unique_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {model_name.upper()}")
    plt.tight_layout()

    os.makedirs("static/confusion", exist_ok=True)
    file_name = f"{uuid.uuid4().hex}_{model_name}_cm.png"
    plt.savefig(f"static/confusion/{file_name}")
    plt.close(fig)
    return f"confusion/{file_name}"


@app.route("/compare_features/<model>/<query_label>/<target_label>")
def compare_features(model, query_label, target_label):
    """Tampilkan grafik perbandingan fitur Query dan Top-K"""
    import uuid

    feat_path = os.path.join("features", f"{model}_aug_features.npy")
    label_path = os.path.join("features", f"{model}_aug_labels.npy")

    features = np.load(feat_path, allow_pickle=True)
    labels = np.load(label_path, allow_pickle=True)

    labels = [os.path.splitext(os.path.basename(l.lower()))[0] for l in labels]
    query_label = os.path.splitext(query_label.lower())[0]
    target_label = os.path.splitext(target_label.lower())[0]

    def find_index(lst, target):
        for i, lbl in enumerate(lst):
            if lbl == target or lbl.startswith(target) or target.startswith(lbl):
                return i
        return None

    q_idx = find_index(labels, query_label)
    t_idx = find_index(labels, target_label)

    if q_idx is None or t_idx is None:
        return f"Label tidak ditemukan dalam fitur {model}.", 404

    feat_q = features[q_idx]
    feat_t = features[t_idx]
    sim = float(cosine_similarity([feat_q], [feat_t])[0][0])

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(feat_q, label="Query", color="blue", alpha=0.7)
    ax.plot(feat_t, label="Top-K", color="red", alpha=0.7)
    ax.set_title(f"{model.upper()} Similarity: {sim:.4f}")
    ax.legend()
    plt.tight_layout()

    os.makedirs("static/feature_plots", exist_ok=True)
    plot_name = f"{uuid.uuid4().hex}_{model}.png"
    plt.savefig(f"static/feature_plots/{plot_name}")
    plt.close(fig)

    return url_for("static", filename=f"feature_plots/{plot_name}")


def compute_map(query_label, retrieved_labels, all_labels, k=5):
    """Hitung mAP sederhana"""
    query_prefix = query_label[:7].lower()
    y_true = [1 if lbl[:7].lower() == query_prefix else 0 for lbl in all_labels]
    y_score = [1 if lbl in retrieved_labels[:k] else 0 for lbl in all_labels]
    try:
        return average_precision_score(y_true, y_score)
    except ValueError:
        return 0.0


# ============================================================
# 🌐 Route Utama
# ============================================================
@app.route("/", methods=["GET", "POST"])
def index():
    retrieval_data = None
    query_name = None
    query_image_url = None

    if request.method == "POST":
        file = request.files.get("query_image")
        if file:
            query_path = os.path.join(TEST_DIR, file.filename)
            file.save(query_path)

            static_query_dir = os.path.join(app.root_path, "static", "query")
            os.makedirs(static_query_dir, exist_ok=True)
            query_copy_path = os.path.join(static_query_dir, file.filename)
            shutil.copy(query_path, query_copy_path)
            query_image_url = url_for("static", filename=f"query/{file.filename}")

            retrieval_data = retrieve_results(query_path)
            query_name = file.filename

    return render_template("results.html",
                           retrieval_data=retrieval_data,
                           query_name=query_name,
                           query_image=query_image_url)


if __name__ == "__main__":
    app.run(debug=True)
