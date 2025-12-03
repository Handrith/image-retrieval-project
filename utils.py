def generate_scatter_plot(features, labels, query_label, model_name):
    """Scatter plot PCA fitur query vs gallery (debug version)."""
    import numpy as np
    import matplotlib.pyplot as plt
    import os, uuid
    from sklearn.decomposition import PCA

    print(f"\n[DEBUG] Mulai generate scatter plot untuk model {model_name}")
    print(f"[DEBUG] features.shape = {getattr(features, 'shape', None)}")
    print(f"[DEBUG] len(labels) = {len(labels)}")
    print(f"[DEBUG] query_label = {query_label}")

    os.makedirs("static/scatter", exist_ok=True)

    if features is None or len(features) == 0:
        print(f"[ERROR] features kosong untuk {model_name}")
        return None
    if len(features) != len(labels):
        print(f"[WARNING] Jumlah features ({len(features)}) != labels ({len(labels)})")

    # --- Sampling agar tidak overload
    max_samples = 2000
    if len(features) > max_samples:
        idx = np.random.choice(len(features), max_samples, replace=False)
        features = features[idx]
        labels = np.array(labels)[idx]

    # --- PCA 2D
    try:
        pca = PCA(n_components=2, random_state=42)
        reduced = pca.fit_transform(features)
        print(f"[DEBUG] PCA berhasil. reduced.shape = {reduced.shape}")
    except Exception as e:
        print(f"[ERROR] PCA gagal: {e}")
        return None

    # --- Scatter plot
    query_label = query_label.lower()
    fig, ax = plt.subplots(figsize=(6, 5))
    for lbl, xy in zip(labels, reduced):
        color = "red" if lbl.lower().startswith(query_label[:7]) else "gray"
        ax.scatter(xy[0], xy[1], color=color, alpha=0.6, s=35)
    ax.set_title(f"Feature Scatter - {model_name.upper()} ({query_label})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.tight_layout()

    file_name = f"{uuid.uuid4().hex}_{model_name}_scatter.png"
    file_path = os.path.join("static/scatter", file_name)
    plt.savefig(file_path)
    plt.close(fig)

    print(f"[DEBUG] Scatter plot tersimpan di: {file_path}")
    return f"scatter/{file_name}"
