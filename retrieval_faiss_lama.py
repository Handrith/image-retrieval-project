import numpy as np
import faiss
import os
from tensorflow.keras.applications import resnet50, mobilenet_v2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_pre
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_pre

# === CONFIG PATH ===
INDEX_DIR = r"C:\ImageRetrieval\faiss_index"
DATASET_DIR = r"C:\ImageRetrieval\dataset"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")

# === LOAD MODEL ===
resnet_model = resnet50.ResNet50(weights='imagenet', include_top=False, pooling='avg')
mobilenet_model = mobilenet_v2.MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

def extract_feature(img_path, model_name="resnet"):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    if model_name == "resnet":
        x = resnet_pre(x)
        feat = resnet_model.predict(x)
    else:
        x = mobilenet_pre(x)
        feat = mobilenet_model.predict(x)

    return feat.flatten().astype('float32')

def retrieve_topk(query_path, top_k=5, model_name="resnet"):
    index_path = os.path.join(INDEX_DIR, f"{model_name}_index.faiss")
    label_path = os.path.join(INDEX_DIR, f"{model_name}_faiss_orig_labels.npy")

    if not (os.path.exists(index_path) and os.path.exists(label_path)):
        raise FileNotFoundError("Index atau label FAISS belum dibuat!")

    index = faiss.read_index(index_path)
    labels = np.load(label_path, allow_pickle=True)

    query_feat = extract_feature(query_path, model_name)
    query_feat = np.expand_dims(query_feat, axis=0)
    distances, indices = index.search(query_feat, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        img_relpath = labels[idx]
        img_path = os.path.join(TRAIN_DIR, os.path.basename(img_relpath))
        results.append({
            "rank": i + 1,
            "distance": float(distances[0][i]),
            "image_path": os.path.relpath(img_path, DATASET_DIR)
        })
    return results
