import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import plotly.express as px


# =========================================================
# CONFIG
# =========================================================

MODELS = [
    "resnet50",
    "clip",
    "dinov2"
]

USE_TSNE = True
USE_UMAP = True
USE_PCA = True

# IMPORTANT:
# Preserve temporal trajectories instead of random points
MAX_IDENTITIES_PER_SEQUENCE = 5

RANDOM_SEED = 42


# =========================================================
# LOAD EMBEDDINGS
# =========================================================

def load_model_embeddings(
    model_dir,
    crops_root,
    max_identities_per_sequence=5
):

    all_embeddings = []
    all_metadata = []

    sequence_dirs = sorted(
        glob.glob(os.path.join(model_dir, "dancetrack*"))
    )

    np.random.seed(RANDOM_SEED)

    for sequence_dir in tqdm(sequence_dirs):

        sequence_name = os.path.basename(sequence_dir)

        identity_files = sorted(
            glob.glob(os.path.join(sequence_dir, "*.npz"))
        )

        # -------------------------------------------------
        # SAMPLE IDENTITIES
        # -------------------------------------------------

        if len(identity_files) > max_identities_per_sequence:

            identity_files = list(
                np.random.choice(
                    identity_files,
                    max_identities_per_sequence,
                    replace=False
                )
            )

        for identity_file in identity_files:

            identity_name = (
                os.path.basename(identity_file)
                .replace(".npz", "")
            )

            data = np.load(identity_file)

            embeddings = data["embeddings"]
            frame_ids = data["frame_ids"]

            for emb, frame_id in zip(embeddings, frame_ids):

                frame_name = f"frame_{frame_id:06d}.jpg"

                image_path = os.path.join(
                    crops_root,
                    sequence_name,
                    identity_name,
                    frame_name
                )

                # -----------------------------------------
                # MEMORY OPTIMIZATION
                # -----------------------------------------

                emb = emb.astype(np.float16)

                all_embeddings.append(emb)

                all_metadata.append({
                    "sequence_id": sequence_name,
                    "identity_id": identity_name,
                    "frame_id": int(frame_id),
                    "image_path": image_path
                })

    embeddings = np.stack(all_embeddings)

    metadata_df = pd.DataFrame(all_metadata)

    return embeddings, metadata_df


# =========================================================
# UMAP
# =========================================================

def run_umap(embeddings):

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.1,
        metric="cosine",
        random_state=RANDOM_SEED
    )

    reduced = reducer.fit_transform(embeddings)

    return reduced


# =========================================================
# TSNE
# =========================================================

def run_tsne(embeddings):

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        metric="cosine",
        random_state=RANDOM_SEED,
        init="pca"
    )

    reduced = tsne.fit_transform(embeddings)

    return reduced


# =========================================================
# PCA
# =========================================================

def run_pca(embeddings):

    pca = PCA(n_components=2)

    reduced = pca.fit_transform(embeddings)

    return reduced


# =========================================================
# PLOT
# =========================================================

def save_plot(
    reduced,
    metadata_df,
    method_name,
    model_name,
    save_dir
):

    plot_df = metadata_df.copy()

    plot_df["x"] = reduced[:, 0]
    plot_df["y"] = reduced[:, 1]

    fig = px.scatter(
        plot_df,
        x="x",
        y="y",
        color="sequence_id",

        hover_data={
            "sequence_id": True,
            "identity_id": True,
            "frame_id": True,
            "image_path": True,
            "x": False,
            "y": False
        },

        title=f"{model_name.upper()} - {method_name.upper()}",
    )

    fig.update_traces(
        marker=dict(
            size=5,
            opacity=0.75
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=900,
        width=1200,
        legend_title="Sequence",

        hoverlabel=dict(
            font_size=12
        )
    )

    html_save_path = os.path.join(
        save_dir,
        f"{model_name}_{method_name}.html"
    )

    fig.write_html(
        html_save_path,
        include_plotlyjs="cdn"
    )

    print(f"\nSaved plot: {html_save_path}")


# =========================================================
# MAIN ANALYSIS
# =========================================================

def analyze_model(
    model_name,
    embeddings_root,
    results_root,
    crops_root
):

    print("\n" + "=" * 60)
    print(f"MANIFOLD ANALYSIS: {model_name}")
    print("=" * 60)

    model_dir = os.path.join(
        embeddings_root,
        model_name
    )

    embeddings, metadata_df = load_model_embeddings(
        model_dir,
        crops_root,
        MAX_IDENTITIES_PER_SEQUENCE
    )

    print(f"\nLoaded embeddings: {len(embeddings)}")

    manifold_dir = os.path.join(
        results_root,
        "manifold",
        model_name
    )

    os.makedirs(manifold_dir, exist_ok=True)

    # -----------------------------------------------------
    # UMAP
    # -----------------------------------------------------

    if USE_UMAP:

        print("\nRunning UMAP...")

        umap_embeddings = run_umap(embeddings)

        save_plot(
            umap_embeddings,
            metadata_df,
            "umap",
            model_name,
            manifold_dir
        )

    # -----------------------------------------------------
    # TSNE
    # -----------------------------------------------------

    if USE_TSNE:

        print("\nRunning t-SNE...")

        tsne_embeddings = run_tsne(embeddings)

        save_plot(
            tsne_embeddings,
            metadata_df,
            "tsne",
            model_name,
            manifold_dir
        )

    # -----------------------------------------------------
    # PCA
    # -----------------------------------------------------

    if USE_PCA:

        print("\nRunning PCA...")

        pca_embeddings = run_pca(embeddings)

        save_plot(
            pca_embeddings,
            metadata_df,
            "pca",
            model_name,
            manifold_dir
        )


# =========================================================
# MAIN
# =========================================================

def main():

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    embeddings_root = os.path.join(
        BASE_DIR,
        "..",
        "..",
        "data",
        "embeddings"
    )

    crops_root = os.path.join(
        BASE_DIR,
        "..",
        "..",
        "data",
        "crops"
    )

    results_root = os.path.join(
        BASE_DIR,
        "..",
        "..",
        "results"
    )

    os.makedirs(results_root, exist_ok=True)

    for model_name in MODELS:

        analyze_model(
            model_name,
            embeddings_root,
            results_root,
            crops_root
        )

    print("\nALL MANIFOLD ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()