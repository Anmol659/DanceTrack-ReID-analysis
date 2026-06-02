import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# CONFIG
# =========================================================

MODELS = [
    "resnet50",
    "clip",
    "dinov2"
]


# =========================================================
# METRICS
# =========================================================

def compute_centroid(embeddings):

    centroid = np.mean(embeddings, axis=0)

    centroid = centroid / np.linalg.norm(centroid)

    return centroid


def compute_compactness(embeddings, centroid):

    similarities = cosine_similarity(
        embeddings,
        centroid.reshape(1, -1)
    ).flatten()

    mean_similarity = similarities.mean()

    variance = similarities.var()

    return mean_similarity, variance, similarities


def compute_temporal_drift(embeddings):

    drifts = []

    for i in range(1, len(embeddings)):

        sim = cosine_similarity(
            embeddings[i].reshape(1, -1),
            embeddings[i - 1].reshape(1, -1)
        )[0][0]

        drift = 1 - sim

        drifts.append(drift)

    if len(drifts) == 0:
        return 0.0, []

    return np.mean(drifts), drifts


# =========================================================
# PROCESS IDENTITY
# =========================================================

def process_identity(npz_path):

    data = np.load(npz_path)

    embeddings = data["embeddings"]

    frame_ids = data["frame_ids"]

    if len(embeddings) < 2:
        return None

    centroid = compute_centroid(embeddings)

    compactness_mean, compactness_var, similarities = (
        compute_compactness(
            embeddings,
            centroid
        )
    )

    drift_mean, drifts = compute_temporal_drift(
        embeddings
    )

    result = {
        "compactness_mean": compactness_mean,
        "compactness_variance": compactness_var,
        "temporal_drift_mean": drift_mean,
        "num_frames": len(frame_ids)
    }

    return result


# =========================================================
# MAIN ANALYSIS
# =========================================================

def analyze_model(model_name, embeddings_root, results_root):

    print("\n" + "=" * 60)
    print(f"ANALYZING MODEL: {model_name}")
    print("=" * 60)

    model_dir = os.path.join(
        embeddings_root,
        model_name
    )

    sequence_dirs = sorted(
        glob.glob(os.path.join(model_dir, "dancetrack*"))
    )

    all_results = []

    for sequence_dir in sequence_dirs:

        sequence_name = os.path.basename(sequence_dir)

        print(f"\nProcessing sequence: {sequence_name}")

        identity_files = sorted(
            glob.glob(os.path.join(sequence_dir, "*.npz"))
        )

        for identity_file in tqdm(identity_files):

            identity_name = os.path.basename(identity_file)

            identity_id = identity_name.replace(".npz", "")

            result = process_identity(identity_file)

            if result is None:
                continue

            result["sequence_id"] = sequence_name
            result["identity_id"] = identity_id
            result["model"] = model_name

            all_results.append(result)

    results_df = pd.DataFrame(all_results)

    save_dir = os.path.join(
        results_root,
        "csv"
    )

    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(
        save_dir,
        f"{model_name}_compactness.csv"
    )

    results_df.to_csv(save_path, index=False)

    print(f"\nSaved results: {save_path}")

    print("\nSummary Statistics:\n")

    print(results_df.describe())

    return results_df


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

    results_root = os.path.join(
        BASE_DIR,
        "..",
        "..",
        "results"
    )

    os.makedirs(results_root, exist_ok=True)

    all_model_results = []

    for model_name in MODELS:

        results_df = analyze_model(
            model_name,
            embeddings_root,
            results_root
        )

        all_model_results.append(results_df)

    combined_df = pd.concat(
        all_model_results,
        ignore_index=True
    )

    combined_save_path = os.path.join(
        results_root,
        "csv",
        "combined_compactness.csv"
    )

    combined_df.to_csv(
        combined_save_path,
        index=False
    )

    print("\nCombined results saved:")
    print(combined_save_path)

    print("\nALL ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()