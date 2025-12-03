import os
import numpy as np
from tqdm import tqdm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

# ================================
# KONFIGURASI
# ================================
DATASET_DIR = "dataset/train"
FEATURES_DIR = "features"
IMG_SIZE = (224, 224)
AUG_MULTIPLIER = 2
MODELS = ["resnet", "mobilenet"]

# ================================
# AUGMENTASI RINGAN
# ================================
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    brightness_range=[0.9, 1.1],
    horizontal_flip=True,
    fill_mode='nearest'
)

# ================================
# LOAD MODEL
# ================================
def load_model(model_name):
    if model_name == "resnet":
        model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        preprocess = resnet_preprocess
    elif model_name == "mobilenet":
        model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
        preprocess = mobilenet_preprocess
    else:
        raise ValueError("Model tidak dikenal.")
    return model, preprocess

# ================================
# DETEKSI FILE BARU
# ================================
def detect_new_images(existing_paths, folder_path):
    """Bandingkan daftar path lama dengan isi folder train."""
    current_files = [os.path.join(folder_path, f)
                     for f in os.listdir(folder_path)
                     if os.path.isfile(os.path.join(folder_path, f))]
    new_files = [f for f in current_files if f not in existing_paths]
    return new_files

# ================================
# EKSTRAKSI FITUR BARU + AUGMENTASI
# ================================
def extract_augmented_features(model, preprocess, new_files, model_name):
    features, labels, img_paths = [], [], []

    if not new_files:
        print(f"✅ Tidak ada file baru untuk {model_name.upper()}.")
        return np.array([]), np.array([]), np.array([])

    print(f"\n📂 Memproses {len(new_files)} file BARU untuk augmentasi ({model_name.upper()})...")

    for fpath in tqdm(new_files, desc=f"Augmentasi {model_name.upper()}"):
        fname = os.path.basename(fpath)
        try:
            img = image.load_img(fpath, target_size=IMG_SIZE)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)

            # Fitur dari gambar asli
            base = preprocess(x.copy())
            feat_base = model.predict(base, verbose=0)
            features.append(feat_base.flatten())
            labels.append(os.path.splitext(fname)[0] + "_orig")
            img_paths.append(fpath)

            # Fitur augmentasi ringan
            aug_iter = datagen.flow(x, batch_size=1)
            for i in range(AUG_MULTIPLIER):
                aug_img = next(aug_iter)[0].astype(np.uint8)
                aug_img = np.expand_dims(aug_img, axis=0)
                aug_img = preprocess(aug_img)
                feat_aug = model.predict(aug_img, verbose=0)
                features.append(feat_aug.flatten())
                labels.append(os.path.splitext(fname)[0] + f"_aug{i+1}")
                img_paths.append(fpath)
        except Exception as e:
            print(f"❌ Gagal memproses {fpath}: {e}")

    return np.array(features), np.array(labels), np.array(img_paths)

# ================================
# SIMPAN FITUR UPDATE
# ================================
def append_features(model_name, new_feat, new_labels, new_paths):
    feat_file = os.path.join(FEATURES_DIR, f"{model_name}_aug_features.npy")
    label_file = os.path.join(FEATURES_DIR, f"{model_name}_aug_labels.npy")
    path_file = os.path.join(FEATURES_DIR, f"{model_name}_aug_paths.npy")

    if os.path.exists(feat_file):
        old_feat = np.load(feat_file, allow_pickle=True)
        old_labels = np.load(label_file, allow_pickle=True)
        old_paths = np.load(path_file, allow_pickle=True)
    else:
        old_feat = np.empty((0, new_feat.shape[1])) if len(new_feat) > 0 else np.empty((0,))
        old_labels, old_paths = np.array([]), np.array([])

    updated_feat = np.vstack([old_feat, new_feat]) if len(new_feat) > 0 else old_feat
    updated_labels = np.concatenate([old_labels, new_labels])
    updated_paths = np.concatenate([old_paths, new_paths])

    np.save(feat_file, updated_feat)
    np.save(label_file, updated_labels)
    np.save(path_file, updated_paths)
    print(f"✅ Update fitur {model_name.upper()} berhasil ({len(updated_feat)} total gambar).")

# ================================
# MAIN PROGRAM
# ================================
def main():
    os.makedirs(FEATURES_DIR, exist_ok=True)

    for model_name in MODELS:
        print(f"\n=== Proses update AUGMENTED {model_name.upper()} ===")
        feat_file = os.path.join(FEATURES_DIR, f"{model_name}_aug_paths.npy")
        existing_paths = np.load(feat_file, allow_pickle=True).tolist() if os.path.exists(feat_file) else []
        new_files = detect_new_images(existing_paths, DATASET_DIR)

        if not new_files:
            print(f"✅ Tidak ada gambar baru untuk {model_name.upper()}. Lewati update.")
            continue

        model, preprocess = load_model(model_name)
        new_feat, new_labels, new_paths = extract_augmented_features(model, preprocess, new_files, model_name)
        if len(new_feat) > 0:
            append_features(model_name, new_feat, new_labels, new_paths)

    print("\n🎯 Proses update augmentasi selesai sepenuhnya!")

if __name__ == "__main__":
    main()
