import os
import numpy as np
from tqdm import tqdm
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.preprocessing import image

# === KONFIGURASI ===
DATASET_DIR = "dataset/train"   # lokasi dataset train
FEATURES_DIR = "features"
IMG_SIZE = (224, 224)

def load_model(model_name):
    """Load CNN model sebagai feature extractor"""
    if model_name == "resnet":
        base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        preprocess = resnet_preprocess
    elif model_name == "mobilenet":
        base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
        preprocess = mobilenet_preprocess
    else:
        raise ValueError("Model tidak dikenal. Gunakan 'resnet' atau 'mobilenet'.")
    return base_model, preprocess

def extract_features(model, preprocess, folder_path):
    """Ekstraksi fitur dari semua gambar dalam satu folder"""
    features = []
    labels = []
    img_paths = []

    # Cek apakah folder kosong
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    if len(files) == 0:
        print(f"⚠️ Folder {folder_path} kosong. Tambahkan gambar terlebih dahulu.")
        return np.array([]), np.array([]), np.array([])

    print(f"\n📂 Memproses {len(files)} file di folder: {folder_path}")

    for fname in tqdm(files, desc="Ekstraksi fitur"):
        fpath = os.path.join(folder_path, fname)
        try:
            img = image.load_img(fpath, target_size=IMG_SIZE)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess(x)
            feat = model.predict(x, verbose=0)
            features.append(feat.flatten())
            labels.append(os.path.splitext(fname)[0])  # gunakan nama file sebagai label
            img_paths.append(fpath)
        except Exception as e:
            print(f"❌ Gagal memproses {fpath}: {e}")

    return np.array(features), np.array(labels), np.array(img_paths)

def save_features(model_name, features, labels, img_paths):
    """Simpan hasil ekstraksi fitur ke file .npy"""
    os.makedirs(FEATURES_DIR, exist_ok=True)

    np.save(os.path.join(FEATURES_DIR, f"{model_name}_features.npy"), features)
    np.save(os.path.join(FEATURES_DIR, f"{model_name}_labels.npy"), labels)
    np.save(os.path.join(FEATURES_DIR, f"{model_name}_paths.npy"), img_paths)

    print(f"✅ Fitur {model_name.upper()} tersimpan di folder '{FEATURES_DIR}/'")
    print(f"   Jumlah gambar: {len(features)}")
    print(f"   Dimensi fitur: {features.shape[1] if len(features) > 0 else 0}")

def main():
    """Pipeline utama ekstraksi fitur"""
    for model_name in ["resnet", "mobilenet"]:
        print(f"\n=== Ekstraksi fitur menggunakan {model_name.upper()} ===")
        model, preprocess = load_model(model_name)
        features, labels, img_paths = extract_features(model, preprocess, DATASET_DIR)

        if len(features) == 0:
            print(f"⚠️ Tidak ada fitur yang diekstrak untuk {model_name.upper()}.\n")
            continue

        save_features(model_name, features, labels, img_paths)

    print("\n🎯 Proses ekstraksi fitur selesai sepenuhnya!")

if __name__ == "__main__":
    main()
