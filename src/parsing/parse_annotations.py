import os
import pandas as pd
from configparser import ConfigParser


def parse_seqinfo(seqinfo_path):
    config = ConfigParser()
    config.read(seqinfo_path)

    seq_data = {
        "name": config.get("Sequence", "name"),
        "imDir": config.get("Sequence", "imDir"),
        "frameRate": config.getint("Sequence", "frameRate"),
        "seqLength": config.getint("Sequence", "seqLength"),
        "imWidth": config.getint("Sequence", "imWidth"),
        "imHeight": config.getint("Sequence", "imHeight"),
        "imExt": config.get("Sequence", "imExt"),
    }

    return seq_data


def parse_gt(gt_path, sequence_name, image_dir):

    columns = [
        "frame",
        "id",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "class",
        "visibility"
    ]

    df = pd.read_csv(
        gt_path,
        header=None,
        names=columns
    )

    df["sequence_id"] = sequence_name

    # Generate image paths
    df["image_path"] = df["frame"].apply(
        lambda x: os.path.join(
            image_dir,
            f"{x:08d}.jpg"
        )
    )

    return df


def build_metadata(dataset_root):

    sequence_dirs = sorted(
        [
            os.path.join(dataset_root, d)
            for d in os.listdir(dataset_root)
            if os.path.isdir(os.path.join(dataset_root, d))
        ]
    )

    print("\nFound sequence directories:\n")

    for s in sequence_dirs:
        print(s)

    print("\nTotal:", len(sequence_dirs))

    all_dfs = []

    for seq_dir in sequence_dirs:

        sequence_name = os.path.basename(seq_dir)

        print(f"\nProcessing: {sequence_name}")

        gt_path = os.path.join(seq_dir, "gt", "gt.txt")
        seqinfo_path = os.path.join(seq_dir, "seqinfo.ini")
        image_dir = os.path.join(seq_dir, "img1")

        if not os.path.exists(gt_path):
            print(f"Missing gt.txt for {sequence_name}")
            continue

        if not os.path.exists(seqinfo_path):
            print(f"Missing seqinfo.ini for {sequence_name}")
            continue

        seqinfo = parse_seqinfo(seqinfo_path)

        gt_df = parse_gt(
            gt_path,
            sequence_name,
            image_dir
        )

        # Add sequence metadata
        gt_df["fps"] = seqinfo["frameRate"]
        gt_df["seq_length"] = seqinfo["seqLength"]
        gt_df["im_width"] = seqinfo["imWidth"]
        gt_df["im_height"] = seqinfo["imHeight"]

        all_dfs.append(gt_df)

    if len(all_dfs) == 0:
        raise ValueError(
            "No valid DanceTrack sequences found. "
            "Check dataset structure and paths."
        )

    final_df = pd.concat(all_dfs, ignore_index=True)

    return final_df


if __name__ == "__main__":

    DATASET_ROOT = r"C:\Users\anmol\Desktop\dancetrack\data\raw"

    metadata_df = build_metadata(DATASET_ROOT)

    print("\nMetadata Preview:\n")
    print(metadata_df.head())

    print("\nColumns:\n")
    print(metadata_df.columns)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    save_dir = os.path.join(
        BASE_DIR,
        "..",
        "..",
        "data",
        "metadata"
    )

    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(
        save_dir,
        "annotations.parquet"
    )

    metadata_df.to_parquet(save_path)

    print("\nMetadata saved:")
    print(save_path)

    print("\nTotal annotations:", len(metadata_df))

    print(
        "Total sequences:",
        metadata_df["sequence_id"].nunique()
    )