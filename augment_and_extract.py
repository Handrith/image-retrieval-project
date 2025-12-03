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
AUG_MULTIPLIER = 2   # berapa banyak augmentasi per gambar
MODELS = ["resnet", "mobilenet"]

# ================================
# AUGMENTASI RINGAN
# ================================
datagen = ImageDataGenerator(
    rotation_range=10,         # rotasi kecil
    width_shift_range=0.05,    # geser horizontal kecil
    height_shift_range=0.05,   # geser vertikal kecil
    zoom_range=0.1,            # zoom kecil
    brightness_range=[0.9, 1.1], # variasi pencahayaan ringan
    horizontal_flip=True,      # flip horizontal
    fill_mode='nearest'
)

# ================================
# LOAD MODEL
# ================================
def load_model(model_name):
    if model_name == "resnet":
        model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        preprocess = resnet_preprocess
        feature_dim = 2048
    elif model_name == "mobilenet":
        model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
        preprocess = mobilenet_preprocess
        feature_dim = 1280
    else:
        raise ValueError("Model tidak dikenal.")
    return model, preprocess, feature_dim

# ================================
# EKSTRAKSI FITUR AUGMENTED
# ================================
def extract_augmented_features(model, preprocess, folder_path, model_name):
    features, labels, img_paths = [], [], []
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    print(f"\n📂 Memproses {len(files)} file untuk augmentasi ({model_name.upper()})...")

    for fname in tqdm(files, desc=f"Augmentasi {model_name.upper()}"):
        fpath = os.path.join(folder_path, fname)
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

            # Fitur dari augmentasi ringan
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
# SIMPAN FITUR
# ================================
def save_features(model_name, features, labels, img_paths):
    os.makedirs(FEATURES_DIR, exist_ok=True)
    np.save(os.path.join(FEATURES_DIR, f"{model_name}_aug_features.npy"), features)
    np.save(os.path.join(FEATURES_DIR, f"{model_name}_aug_labels.npy"), labels)
    np.save(os.path.join(FEATURES_DIR, f"{model_name}_aug_paths.npy"), img_paths)
    print(f"✅ Fitur AUGMENTED {model_name.upper()} tersimpan ({len(features)} gambar total)")

# ================================
# MAIN PROGRAM
# ================================
def main():
    for model_name in MODELS:
        model, preprocess, feat_dim = load_model(model_name)
        features, labels, img_paths = extract_augmented_features(model, preprocess, DATASET_DIR, model_name)
        save_features(model_name, features, labels, img_paths)

    print("\n🎯 Proses augmentasi & ekstraksi selesai sepenuhnya!")

if __name__ == "__main__":
    main()
