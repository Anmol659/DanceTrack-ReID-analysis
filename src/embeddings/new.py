import os
import json
import numpy as np
from tqdm import tqdm
import umap.umap_ as umap

# ============================================================
# CONFIG
# ============================================================

# ROOT OF SAVED EMBEDDINGS
EMBEDDINGS_ROOT = r"C:\Users\anmol\Desktop\embedding_explorer\datasets\raw\dancetrack"

# ROOT OF CROPPED IMAGES
CROPS_ROOT = r"C:\Users\anmol\Desktop\dancetrack\data\crops"

# OUTPUT
OUTPUT_ROOT = r"C:\Users\anmol\Desktop\embedding_explorer\datasets\processed\dancetrack"

# ============================================================
# CREATE OUTPUT DIR
# ============================================================

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ============================================================
# SAVE JSON
# ============================================================

def save_json(data, save_path):

    with open(save_path, "w") as f:
        json.dump(data, f)

# ============================================================
# FIND IMAGE PATH
# ============================================================

def find_image_path(sequence_name, track_id, frame_id):

    """
    Searches crop folder for matching frame image.
    """

    track_folder = os.path.join(
        CROPS_ROOT,
        sequence_name,
        f"id_{track_id:04d}"
    )

    if not os.path.exists(track_folder):
        return None

    possible_names = [
        f"{frame_id:06d}.jpg",
        f"{frame_id:06d}.png",
        f"{frame_id}.jpg",
        f"{frame_id}.png"
    ]

    for name in possible_names:

        img_path = os.path.join(track_folder, name)

        if os.path.exists(img_path):
            return img_path

    # fallback search
    for file in os.listdir(track_folder):

        if str(frame_id) in file:
            return os.path.join(track_folder, file)

    return None

# ============================================================
# PROCESS EACH MODEL
# ============================================================

model_dirs = [
    d for d in os.listdir(EMBEDDINGS_ROOT)
    if os.path.isdir(os.path.join(EMBEDDINGS_ROOT, d))
]

for model_name in model_dirs:

    print("\n===================================")
    print(f"PROCESSING MODEL: {model_name}")
    print("===================================")

    model_root = os.path.join(
        EMBEDDINGS_ROOT,
        model_name
    )

    all_embeddings = []
    all_metadata = []

    # --------------------------------------------------------
    # SEQUENCES
    # --------------------------------------------------------

    sequences = [
        d for d in os.listdir(model_root)
        if os.path.isdir(os.path.join(model_root, d))
    ]

    for sequence_name in tqdm(sequences, desc="Sequences"):

        sequence_dir = os.path.join(
            model_root,
            sequence_name
        )

        npz_files = [
            f for f in os.listdir(sequence_dir)
            if f.endswith(".npz")
        ]

        for npz_file in npz_files:

            npz_path = os.path.join(
                sequence_dir,
                npz_file
            )

            try:

                data = np.load(npz_path, allow_pickle=True)

                embeddings = data["embeddings"]
                frame_ids = data["frame_ids"]

                # --------------------------------------------
                # TRACK ID
                # --------------------------------------------

                track_id = int(
                    npz_file.replace("id_", "")
                            .replace(".npz", "")
                )

                # --------------------------------------------
                # STORE
                # --------------------------------------------

                for i in range(len(embeddings)):

                    frame_id = int(frame_ids[i])

                    image_path = find_image_path(
                        sequence_name,
                        track_id,
                        frame_id
                    )

                    all_embeddings.append(embeddings[i])

                    all_metadata.append({
                        "sequence": sequence_name,
                        "track_id": track_id,
                        "frame_id": frame_id,
                        "image": image_path
                    })

            except Exception as e:

                print(f"\nERROR: {npz_path}")
                print(e)

    # ========================================================
    # CONVERT TO ARRAY
    # ========================================================

    all_embeddings = np.array(all_embeddings)

    print(f"\nTotal embeddings: {len(all_embeddings)}")
    print(f"Embedding shape: {all_embeddings.shape}")

    # ========================================================
    # RUN UMAP
    # ========================================================

    print("\nRunning UMAP...")

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42
    )

    coords = reducer.fit_transform(all_embeddings)

    # ========================================================
    # CREATE JSON
    # ========================================================

    print("\nCreating JSON...")

    json_data = []

    for i in range(len(coords)):

        point = {
            "x": float(coords[i][0]),
            "y": float(coords[i][1]),
            "index": i,
            "sequence": all_metadata[i]["sequence"],
            "track_id": all_metadata[i]["track_id"],
            "frame_id": all_metadata[i]["frame_id"],
            "image": all_metadata[i]["image"]
        }

        json_data.append(point)

    # ========================================================
    # SAVE
    # ========================================================

    output_dir = os.path.join(
        OUTPUT_ROOT,
        model_name
    )

    os.makedirs(output_dir, exist_ok=True)

    save_path = os.path.join(
        output_dir,
        "umap.json"
    )

    save_json(json_data, save_path)

    print(f"\nSaved:")
    print(save_path)

print("\nALL DONE")