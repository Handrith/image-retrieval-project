import time
from tqdm import tqdm
import numpy as np

def evaluate_model(model_name, model, query_loader, gallery_features, gallery_labels, similarity_fn):
    print(f"\nMulai evaluasi untuk model: {model_name}")
    start_time = time.time()

    total_queries = len(query_loader)
    correct = 0
    times = []

    for img_path, label in tqdm(query_loader, desc=f"Evaluasi {model_name}", total=total_queries):
        t0 = time.time()

        # ekstraksi fitur query
        query_feature = model.extract_features(img_path)
        similarities = [similarity_fn(query_feature, gf) for gf in gallery_features]
        top_idx = np.argmax(similarities)
        predicted_label = gallery_labels[top_idx]

        if predicted_label == label:
            correct += 1

        times.append(time.time() - t0)

    total_time = time.time() - start_time
    avg_time = np.mean(times)
    accuracy = (correct / total_queries) * 100

    # tampilkan ringkasan di terminal
    print(f"\n=== RINGKASAN EVALUASI: {model_name} ===")
    print(f"Total Query         : {total_queries}")
    print(f"Akurasi             : {accuracy:.2f}%")
    print(f"Total Waktu         : {total_time:.2f} detik")
    print(f"Rata-rata per Query : {avg_time:.3f} detik\n")

    # kembalikan hasil untuk digunakan pada grafik/tabel
    return {
        "model": model_name,
        "total_queries": total_queries,
        "accuracy": accuracy,
        "total_time": total_time,
        "avg_time": avg_time
    }
