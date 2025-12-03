import os
import numpy as np
from tqdm import tqdm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

DATASET_DIR = "dataset/train"
FEATURES_DIR = "features"
IMG_SIZE = (224, 224)

# === Fungsi umum ===
def normalize_path(p):
    return os.path.normpath(p).replace("\\", "/")

def extract_new_features(model, preprocess, existing_paths, model_name):
    existing_paths = [normalize_path(p) for p in existing_paths]
    files = [f for f in os.listdir(DATASET_DIR) if os.path.isfile(os.path.join(DATASET_DIR, f))]

    new_files = []
    for f in files:
        fpath = normalize_path(os.path.join(DATASET_DIR, f))
        if fpath not in existing_paths:
            new_files.append(f)

    if not new_files:
        print(f"✅ Tidak ada gambar baru untuk model {model_name.upper()}.")
        return np.array([]), np.array([]), np.array([])

    print(f"\n📂 {model_name.upper()}: Menemukan {len(new_files)} gambar baru.")
    new_features, new_labels, new_paths = [], [], []
    for fname in tqdm(new_files, desc=f"Ekstraksi {model_name}"):
        fpath = normalize_path(os.path.join(DATASET_DIR, fname))
        try:
            img = image.load_img(fpath, target_size=IMG_SIZE)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess(x)
            feat = model.predict(x, verbose=0)
            new_features.append(feat.flatten())
            new_labels.append(os.path.splitext(fname)[0])
            new_paths.append(fpath)
        except Exception as e:
            print(f"❌ Gagal memproses {fpath}: {e}")

    return np.array(new_features), np.array(new_labels), np.array(new_paths)

def update_model_features(model_name, model, preprocess):
    feat_path = os.path.join(FEATURES_DIR, f"{model_name}_features.npy")
    label_path = os.path.join(FEATURES_DIR, f"{model_name}_labels.npy")
    path_path = os.path.join(FEATURES_DIR, f"{model_name}_paths.npy")

    if not os.path.exists(feat_path):
        print(f"⚠️ File fitur untuk {model_name.upper()} belum ada. Jalankan extract_features.py dulu.")
        return

    old_features = np.load(feat_path)
    old_labels = np.load(label_path)
    old_paths = np.load(path_path)

    new_features, new_labels, new_paths = extract_new_features(model, preprocess, old_paths, model_name)

    if len(new_features) > 0:
        all_features = np.concatenate([old_features, new_features])
        all_labels = np.concatenate([old_labels, new_labels])
        all_paths = np.concatenate([old_paths, new_paths])

        np.save(feat_path, all_features)
        np.save(label_path, all_labels)
        np.save(path_path, all_paths)

        print(f"✅ {model_name.upper()} berhasil diupdate! Total gambar sekarang: {len(all_features)}")
    else:
        print(f"ℹ️ Tidak ada pembaruan fitur untuk {model_name.upper()}.")

def main():
    os.makedirs(FEATURES_DIR, exist_ok=True)

    # RESNET
    resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    update_model_features("resnet", resnet_model, resnet_preprocess)

    # MOBILENET
    mobilenet_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
    update_model_features("mobilenet", mobilenet_model, mobilenet_preprocess)

    print("\n🎯 Update selesai untuk kedua model!")

if __name__ == "__main__":
    main()
