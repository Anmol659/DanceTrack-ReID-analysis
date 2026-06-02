import os
import glob
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

import streamlit as st
from streamlit_plotly_events import plotly_events

import plotly.express as px

import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="DanceTrack Embedding Explorer",
    layout="wide"
)

MODELS = [
    "resnet50",
    "clip",
    "dinov2"
]

METHODS = [
    "umap",
    "tsne",
    "pca"
]

MAX_IDENTITIES_PER_SEQUENCE = 5

RANDOM_SEED = 42

CACHE_DIR = "cache"

os.makedirs(CACHE_DIR, exist_ok=True)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------
# ABSOLUTE PATHS
# ---------------------------------------------------------

EMBEDDINGS_ROOT = os.path.join(
    BASE_DIR,
    "data",
    "embeddings"
)

CROPS_ROOT = os.path.join(
    BASE_DIR,
    "data",
    "crops"
)

# ---------------------------------------------------------
# DEBUG
# ---------------------------------------------------------

print("\nEMBEDDINGS ROOT:")
print(EMBEDDINGS_ROOT)

print("\nCROPS ROOT:")
print(CROPS_ROOT)

print("\nEMBEDDINGS EXISTS:")
print(os.path.exists(EMBEDDINGS_ROOT))

print("\nCROPS EXISTS:")
print(os.path.exists(CROPS_ROOT))

# =========================================================
# LOAD EMBEDDINGS
# =========================================================

@st.cache_data(show_spinner=False)
def load_model_embeddings(
    model_name,
    max_identities_per_sequence=5
):

    model_dir = os.path.join(
        EMBEDDINGS_ROOT,
        model_name
    )

    all_embeddings = []
    all_metadata = []

    sequence_dirs = sorted(
        glob.glob(os.path.join(model_dir, "dancetrack*"))
    )

    np.random.seed(RANDOM_SEED)

    for sequence_dir in tqdm(sequence_dirs):
        print("\nSEQUENCE:")
        print(sequence_dir)

        sequence_name = os.path.basename(sequence_dir)

        identity_files = sorted(
            print("\nFOUND IDENTITY FILES:"),
            print(len(identity_files)),
            glob.glob(os.path.join(sequence_dir, "*.npz"))
        )

        # ---------------------------------------------
        # SAMPLE IDENTITIES
        # ---------------------------------------------

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
                    CROPS_ROOT,
                    sequence_name,
                    identity_name,
                    frame_name
                )

                emb = emb.astype(np.float16)

                all_embeddings.append(emb)

                all_metadata.append({
                    "sequence_id": sequence_name,
                    "identity_id": identity_name,
                    "frame_id": int(frame_id),
                    "image_path": image_path
                })

    if len(all_embeddings) == 0:

     raise ValueError(
        f"\nNo embeddings found for model: {model_name}\n"
        f"Checked path: {model_dir}\n"
        f"Verify embedding extraction output."
    )

    embeddings = np.stack(all_embeddings)

    metadata_df = pd.DataFrame(all_metadata)

    return embeddings, metadata_df


# =========================================================
# DIMENSIONALITY REDUCTION
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


def run_pca(embeddings):

    pca = PCA(n_components=2)

    reduced = pca.fit_transform(embeddings)

    return reduced


# =========================================================
# PRECOMPUTE + CACHE
# =========================================================

def get_cache_path(model_name, method_name):

    return os.path.join(
        CACHE_DIR,
        f"{model_name}_{method_name}.pkl"
    )


def precompute_manifold(model_name, method_name):

    cache_path = get_cache_path(
        model_name,
        method_name
    )

    # -----------------------------------------------------
    # LOAD CACHE
    # -----------------------------------------------------

    if os.path.exists(cache_path):

        with open(cache_path, "rb") as f:
            plot_df = pickle.load(f)

        return plot_df

    # -----------------------------------------------------
    # COMPUTE
    # -----------------------------------------------------

    print(f"Computing {model_name} - {method_name}")

    embeddings, metadata_df = load_model_embeddings(
        model_name,
        MAX_IDENTITIES_PER_SEQUENCE
    )

    if method_name == "umap":

        reduced = run_umap(embeddings)

    elif method_name == "tsne":

        reduced = run_tsne(embeddings)

    elif method_name == "pca":

        reduced = run_pca(embeddings)

    else:
        raise ValueError("Invalid method")

    plot_df = metadata_df.copy()

    plot_df["x"] = reduced[:, 0]
    plot_df["y"] = reduced[:, 1]

    # -----------------------------------------------------
    # SAVE CACHE
    # -----------------------------------------------------

    with open(cache_path, "wb") as f:
        pickle.dump(plot_df, f)

    return plot_df


# =========================================================
# PRELOAD ALL
# =========================================================

@st.cache_resource(show_spinner=True)
def preload_all_manifolds():

    all_data = {}

    for model_name in MODELS:

        for method_name in METHODS:

            key = f"{model_name}_{method_name}"

            plot_df = precompute_manifold(
                model_name,
                method_name
            )

            all_data[key] = plot_df

    return all_data


# =========================================================
# LOAD PRECOMPUTED DATA
# =========================================================

with st.spinner("Loading precomputed manifolds..."):

    ALL_MANIFOLDS = preload_all_manifolds()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("DanceTrack Explorer")

selected_model = st.sidebar.selectbox(
    "Model",
    MODELS
)

selected_method = st.sidebar.selectbox(
    "Projection",
    METHODS
)

key = f"{selected_model}_{selected_method}"

plot_df = ALL_MANIFOLDS[key]


# =========================================================
# PLOT
# =========================================================

st.title("DanceTrack Semantic Embedding Explorer")

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

    title=f"{selected_model.upper()} - {selected_method.upper()}"
)

fig.update_traces(
    marker=dict(
        size=6,
        opacity=0.75
    )
)

fig.update_layout(
    template="plotly_dark",
    height=800
)

selected_points = plotly_events(
    fig,
    click_event=True,
    hover_event=False,
    select_event=False,
    override_height=800
)


# =========================================================
# CLICKED POINT
# =========================================================

if selected_points:

    point_index = selected_points[0]["pointIndex"]

    selected_row = plot_df.iloc[point_index]

    st.subheader("Selected Embedding")

    col1, col2 = st.columns([1, 2])

    with col1:

        image_path = selected_row["image_path"]

        if os.path.exists(image_path):

            st.image(
                image_path,
                caption=os.path.basename(image_path),
                use_container_width=True
            )

        else:
            st.error("Image not found")

    with col2:

        st.write("### Metadata")

        st.write(
            {
                "Sequence": selected_row["sequence_id"],
                "Identity": selected_row["identity_id"],
                "Frame": int(selected_row["frame_id"]),
                "Image Path": selected_row["image_path"]
            }
        )


# =========================================================
# TEMPORAL TRAJECTORY
# =========================================================

st.subheader("Temporal Trajectory")

trajectory_identity = st.selectbox(
    "Select Identity",
    sorted(plot_df["identity_id"].unique())
)

trajectory_df = plot_df[
    plot_df["identity_id"] == trajectory_identity
].sort_values("frame_id")

trajectory_fig = px.line(
    trajectory_df,
    x="x",
    y="y",
    markers=True,
    hover_data=[
        "frame_id",
        "sequence_id"
    ],
    title=f"Temporal Trajectory - {trajectory_identity}"
)

trajectory_fig.update_layout(
    template="plotly_dark",
    height=600
)

st.plotly_chart(
    trajectory_fig,
    use_container_width=True
)