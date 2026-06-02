import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# PATHS
# ============================================================

EMBEDDING_ROOT = r"C:\Users\anmol\Desktop\dancetrack\data\embeddings"

OUTPUT_DIR = r"C:\Users\anmol\Desktop\dancetrack\results\plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# HELPERS
# ============================================================

def l2_normalize(x):
    norm = np.linalg.norm(x)
    if norm == 0:
        return x
    return x / norm


def mean_identity_embedding(npz_path):
    """
    Compute representative embedding for one identity.
    """

    data = np.load(npz_path)

    emb = data["embeddings"]

    mean_emb = emb.mean(axis=0)

    return l2_normalize(mean_emb)


def sequence_similarity(sequence_dir):
    """
    Mean inter-person cosine similarity for one sequence.
    """

    identity_vectors = []

    files = sorted(
        f for f in os.listdir(sequence_dir)
        if f.endswith(".npz")
    )

    for f in files:

        path = os.path.join(sequence_dir, f)

        identity_vectors.append(
            mean_identity_embedding(path)
        )

    identity_vectors = np.vstack(identity_vectors)

    n_ids = len(identity_vectors)

    if n_ids < 2:
        return np.nan

    sim_matrix = cosine_similarity(identity_vectors)

    mask = ~np.eye(n_ids, dtype=bool)

    mean_similarity = sim_matrix[mask].mean()

    return mean_similarity


# ============================================================
# MAIN
# ============================================================

models = sorted(
    d for d in os.listdir(EMBEDDING_ROOT)
    if os.path.isdir(os.path.join(EMBEDDING_ROOT, d))
)

print("\nFound models:")
print(models)

for model in models:

    print("\n" + "=" * 70)
    print(model.upper())
    print("=" * 70)

    model_dir = os.path.join(
        EMBEDDING_ROOT,
        model
    )

    sequence_names = []
    similarity_values = []

    sequences = sorted(
        d for d in os.listdir(model_dir)
        if os.path.isdir(os.path.join(model_dir, d))
    )

    for seq in sequences:

        seq_dir = os.path.join(model_dir, seq)

        try:

            score = sequence_similarity(seq_dir)

            sequence_names.append(seq)
            similarity_values.append(score)

            print(
                f"{seq:<20} "
                f"similarity = {score:.4f}"
            )

        except Exception as e:

            print(
                f"Failed {seq}: {e}"
            )

    # ========================================================
    # PLOT
    # ========================================================

    plt.figure(figsize=(16, 6))

    x = np.arange(len(sequence_names))

    plt.bar(x, similarity_values)

    mean_score = np.nanmean(similarity_values)

    plt.axhline(
        mean_score,
        linestyle="--",
        linewidth=2,
        label=f"Dataset Mean = {mean_score:.3f}"
    )

    plt.xticks(
        x,
        sequence_names,
        rotation=90
    )

    plt.ylabel(
        "Mean Inter-Person Cosine Similarity"
    )

    plt.xlabel(
        "DanceTrack Sequence"
    )

    plt.title(
        f"{model.upper()} : Appearance Ambiguity Across Sequences"
    )

    plt.legend()

    plt.tight_layout()

    save_path = os.path.join(
        OUTPUT_DIR,
        f"{model}_inter_person_similarity.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nSaved: {save_path}"
    )

print("\nDone.")