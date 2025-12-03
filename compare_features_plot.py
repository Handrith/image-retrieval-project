import numpy as np
import matplotlib.pyplot as plt
import os

# ================================
# KONFIGURASI
# ================================
FEATURES_DIR = "features"
MODEL_NAME = "resnet"   # atau "mobilenet"
QUERY_NAME = "ANBL3-NB08032-C04 SILV-68.1.JPG"
TOP1_NAME = "ANBL3-NB08050-C08-SIL-68.jpg"

# ================================
# MEMUAT FITUR
# ================================
feat_path = os.path.join(FEATURES_DIR, f"{MODEL_NAME}_aug_features.npy")
label_path = os.path.join(FEATURES_DIR, f"{MODEL_NAME}_aug_labels.npy")

features = np.load(feat_path, allow_pickle=True)
labels = np.load(label_path, allow_pickle=True)

# Pastikan nama file tanpa path
labels = [os.path.basename(l) for l in labels]

# Ambil fitur query dan top-1
query_idx = labels.index(QUERY_NAME)
top1_idx = labels.index(TOP1_NAME)

query_feat = features[query_idx]
top1_feat = features[top1_idx]

# ================================
# HITUNG SIMILARITY
# ================================
similarity = np.dot(query_feat, top1_feat) / (np.linalg.norm(query_feat) * np.linalg.norm(top1_feat))
print(f"🔍 Similarity antara {QUERY_NAME} dan {TOP1_NAME}: {similarity:.4f}")

# ================================
# VISUALISASI FITUR
# ================================
plt.figure(figsize=(12, 6))
plt.plot(query_feat, label=f"Query: {QUERY_NAME}", color='blue', alpha=0.7)
plt.plot(top1_feat, label=f"Top-1: {TOP1_NAME} (sim={similarity:.4f})", color='orange', alpha=0.7)

plt.title(f"Perbandingan Vektor Fitur ({MODEL_NAME.upper()})")
plt.xlabel("Indeks Fitur")
plt.ylabel("Nilai Fitur")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
