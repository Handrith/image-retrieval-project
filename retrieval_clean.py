import os
import numpy as np
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# === KONFIGURASI ===
DATASET_TRAIN = "dataset/train"
DATASET_TEST = "dataset/test"
FEATURES_DIR = "features"
IMG_SIZE = (224, 224)
TOP_N = 5

# === PILIHAN MODE ===
USE_AUGMENTED = True  # ubah ke True jika ingin pakai fitur augmentasi
MODEL_NAME = "resnet"  # pilihan: "resnet" atau "mobilenet"
SHOW_ONLY_ORIG = True  # tampilkan hanya file dengan _orig

# === MUAT MODEL ===
def load_model(model_name):
    if model_name == "resnet":
        model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        preprocess = resnet_preprocess
    elif model_name == "mobilenet":
        model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
        preprocess = mobilenet_preprocess
    else:
        raise ValueError("Model tidak dikenal. Gunakan 'resnet' atau 'mobilenet'.")
    return model, preprocess

# === MUAT FITUR ===
def load_features(model_name, augmented=False):
    suffix = "_aug" if augmented else ""
    try:
        features = np.load(os.path.join(FEATURES_DIR, f"{model_name}{suffix}_features.npy"))
        labels = np.load(os.path.join(FEATURES_DIR, f"{model_name}{suffix}_labels.npy"))
        paths = np.load(os.path.join(FEATURES_DIR, f"{model_name}{suffix}_paths.npy"))
        print(f"✅ Memuat fitur {model_name.upper()}{' (AUGMENTED)' if augmented else ''}")
        return features, labels, paths
    except Exception as e:
        print(f"❌ Gagal memuat fitur: {e}")
        return None, None, None

# === EKSTRAKSI FITUR QUERY ===
def extract_query_feature(model, preprocess, img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess(x)
    feat = model.predict(x, verbose=0)
    return feat.flatten()

# === TEMUKAN KEMIRIPAN ===
def find_similar_images(query_feat, db_features, db_labels, db_paths, top_n=5, only_orig=False):
    sims = cosine_similarity([query_feat], db_features)[0]
    top_idx = np.argsort(sims)[::-1]
    results = [(db_labels[i], db_paths[i], sims[i]) for i in top_idx]

    # 🔹 Filter hanya label dengan "_orig"
    if only_orig:
        results = [r for r in results if r[0].endswith("_orig")]

    # Ambil hanya top-N hasil
    return results[:top_n]

# === VISUALISASI HASIL ===
def show_results(query_path, results):
    if len(results) == 0:
        print("⚠️ Tidak ada hasil yang cocok (setelah filter).")
        return

    plt.figure(figsize=(12, 4))
    plt.subplot(1, len(results)+1, 1)
    plt.imshow(image.load_img(query_path))
    plt.title("Query")
    plt.axis("off")

    for i, (label, path, sim) in enumerate(results):
        plt.subplot(1, len(results)+1, i+2)
        plt.imshow(image.load_img(path))
        plt.title(f"{label}\nSim: {sim:.3f}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# === MAIN ===
def main():
    print(f"\n🔍 Retrieval dengan {MODEL_NAME.upper()} {'(AUGMENTED)' if USE_AUGMENTED else '(BASELINE)'}")
    print(f"   Menampilkan hanya hasil dengan '_orig': {SHOW_ONLY_ORIG}")

    model, preprocess = load_model(MODEL_NAME)
    db_features, db_labels, db_paths = load_features(MODEL_NAME, USE_AUGMENTED)

    if db_features is None or len(db_features) == 0:
        print("⚠️ Tidak ada fitur yang dimuat. Pastikan sudah menjalankan ekstraksi.")
        return

    query_files = [f for f in os.listdir(DATASET_TEST) if os.path.isfile(os.path.join(DATASET_TEST, f))]
    if len(query_files) == 0:
        print(f"⚠️ Folder {DATASET_TEST} kosong.")
        return

    for qf in query_files:
        query_path = os.path.join(DATASET_TEST, qf)
        print(f"\n🔸 Query: {qf}")
        query_feat = extract_query_feature(model, preprocess, query_path)

        results = find_similar_images(
            query_feat,
            db_features,
            db_labels,
            db_paths,
            TOP_N,
            only_orig=SHOW_ONLY_ORIG
        )

        if len(results) == 0:
            print("⚠️ Tidak ada hasil yang cocok dengan filter '_orig'.")
        else:
            for rank, (label, path, sim) in enumerate(results, start=1):
                print(f"{rank}. {label} | Similarity = {sim:.3f}")

        show_results(query_path, results)

    print("\n🎯 Retrieval selesai.")

if __name__ == "__main__":
    main()
