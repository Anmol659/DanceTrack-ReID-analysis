import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.models as models
import torchvision.transforms as transforms

import open_clip
from transformers import AutoImageProcessor, AutoModel


# =========================================================
# CONFIG
# =========================================================

BATCH_SIZE = 64
NUM_WORKERS = 0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

USE_FP16 = DEVICE == "cuda"

# Models to run
MODELS_TO_RUN = [
    "resnet50",
    "clip",
    "dinov2"
]

# =========================================================
# DATASET
# =========================================================

class CropDataset(Dataset):

    def __init__(self, image_paths, preprocess):
        self.image_paths = image_paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        image_path = self.image_paths[idx]

        image = Image.open(image_path).convert("RGB")

        image = self.preprocess(image)

        frame_id = int(
            os.path.basename(image_path)
            .replace("frame_", "")
            .replace(".jpg", "")
        )

        return image, frame_id


# =========================================================
# MODEL LOADERS
# =========================================================

def load_resnet50():

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # Remove classifier
    model = torch.nn.Sequential(*list(model.children())[:-1])

    model = model.to(DEVICE)
    model.eval()

    preprocess = models.ResNet50_Weights.IMAGENET1K_V2.transforms()

    return model, preprocess


def load_clip():

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="openai"
    )

    model = model.to(DEVICE)
    model.eval()

    return model, preprocess


def load_dinov2():

    processor = AutoImageProcessor.from_pretrained(
        "facebook/dinov2-base"
    )

    model = AutoModel.from_pretrained(
        "facebook/dinov2-base"
    )

    model = model.to(DEVICE)
    model.eval()

    return model, processor


# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_resnet_embeddings(model, dataloader):

    all_embeddings = []
    all_frame_ids = []

    with torch.no_grad():

        for images, frame_ids in tqdm(dataloader):

            images = images.to(DEVICE)

            with torch.cuda.amp.autocast(enabled=USE_FP16):

                features = model(images)

                features = features.squeeze(-1).squeeze(-1)

                features = F.normalize(features, dim=-1)

            features = features.cpu().numpy()

            all_embeddings.append(features)

            all_frame_ids.extend(frame_ids.numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)

    return all_embeddings, np.array(all_frame_ids)


def extract_clip_embeddings(model, dataloader):

    all_embeddings = []
    all_frame_ids = []

    with torch.no_grad():

        for images, frame_ids in tqdm(dataloader):

            images = images.to(DEVICE)

            with torch.cuda.amp.autocast(enabled=USE_FP16):

                features = model.encode_image(images)

                features = F.normalize(features, dim=-1)

            features = features.cpu().numpy()

            all_embeddings.append(features)

            all_frame_ids.extend(frame_ids.numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)

    return all_embeddings, np.array(all_frame_ids)


def extract_dinov2_embeddings(model, processor, image_paths):

    all_embeddings = []
    all_frame_ids = []

    batch_size = BATCH_SIZE

    with torch.no_grad():

        for i in tqdm(range(0, len(image_paths), batch_size)):

            batch_paths = image_paths[i:i + batch_size]

            images = [
                Image.open(p).convert("RGB")
                for p in batch_paths
            ]

            frame_ids = [
                int(
                    os.path.basename(p)
                    .replace("frame_", "")
                    .replace(".jpg", "")
                )
                for p in batch_paths
            ]

            inputs = processor(
                images=images,
                return_tensors="pt"
            )

            inputs = {
                k: v.to(DEVICE)
                for k, v in inputs.items()
            }

            with torch.cuda.amp.autocast(enabled=USE_FP16):

                outputs = model(**inputs)

                features = outputs.last_hidden_state[:, 0]

                features = F.normalize(features, dim=-1)

            features = features.cpu().numpy()

            all_embeddings.append(features)

            all_frame_ids.extend(frame_ids)

    all_embeddings = np.concatenate(all_embeddings, axis=0)

    return all_embeddings, np.array(all_frame_ids)


# =========================================================
# PROCESS IDENTITY
# =========================================================

def process_identity(
    model_name,
    model,
    preprocess,
    identity_dir,
    save_path
):

    image_paths = sorted(
        glob.glob(os.path.join(identity_dir, "*.jpg"))
    )

    if len(image_paths) == 0:
        return

    if model_name == "dinov2":

        embeddings, frame_ids = extract_dinov2_embeddings(
            model,
            preprocess,
            image_paths
        )

    else:

        dataset = CropDataset(
            image_paths,
            preprocess
        )

        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True
        )

        if model_name == "resnet50":

            embeddings, frame_ids = extract_resnet_embeddings(
                model,
                dataloader
            )

        elif model_name == "clip":

            embeddings, frame_ids = extract_clip_embeddings(
                model,
                dataloader
            )

    np.savez_compressed(
        save_path,
        embeddings=embeddings,
        frame_ids=frame_ids
    )


# =========================================================
# MAIN
# =========================================================

def main():

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    crops_root = os.path.join(
        BASE_DIR,
        "..",
        "..",
        "data",
        "crops"
    )

    embeddings_root = os.path.join(
        BASE_DIR,
        "..",
        "..",
        "data",
        "embeddings"
    )

    os.makedirs(embeddings_root, exist_ok=True)

    sequence_dirs = sorted(
        glob.glob(os.path.join(crops_root, "dancetrack*"))
    )

    print(f"\nFound sequences: {len(sequence_dirs)}")

    for model_name in MODELS_TO_RUN:

        print("\n" + "=" * 60)
        print(f"RUNNING MODEL: {model_name}")
        print("=" * 60)

        # -------------------------------------------------
        # LOAD MODEL
        # -------------------------------------------------

        if model_name == "resnet50":

            model, preprocess = load_resnet50()

        elif model_name == "clip":

            model, preprocess = load_clip()

        elif model_name == "dinov2":

            model, preprocess = load_dinov2()

        else:
            continue

        # -------------------------------------------------
        # OUTPUT DIR
        # -------------------------------------------------

        model_output_root = os.path.join(
            embeddings_root,
            model_name
        )

        os.makedirs(model_output_root, exist_ok=True)

        # -------------------------------------------------
        # PROCESS SEQUENCES
        # -------------------------------------------------

        for sequence_dir in sequence_dirs:

            sequence_name = os.path.basename(sequence_dir)

            print(f"\nProcessing sequence: {sequence_name}")

            sequence_output_dir = os.path.join(
                model_output_root,
                sequence_name
            )

            os.makedirs(sequence_output_dir, exist_ok=True)

            identity_dirs = sorted(
                glob.glob(os.path.join(sequence_dir, "id_*"))
            )

            for identity_dir in tqdm(identity_dirs):

                identity_name = os.path.basename(identity_dir)

                save_path = os.path.join(
                    sequence_output_dir,
                    f"{identity_name}.npz"
                )

                if os.path.exists(save_path):
                    continue

                process_identity(
                    model_name,
                    model,
                    preprocess,
                    identity_dir,
                    save_path
                )

        print(f"\nCompleted: {model_name}")

    print("\nAll embedding extraction complete.")


if __name__ == "__main__":
    main()